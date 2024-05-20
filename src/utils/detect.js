import * as tf from "@tensorflow/tfjs";
import { renderPolygons } from "./renderBox";
import labels from "./labels.json";

const numClass = labels.length;

/**
 * Preprocess image / frame before forwarded into the model
 * @param {HTMLVideoElement|HTMLImageElement} source
 * @param {Number} modelWidth
 * @param {Number} modelHeight
 * @returns input tensor, xRatio and yRatio
 */
const preprocess = (source, modelWidth, modelHeight) => {
  let xRatio, yRatio; // ratios for boxes

  const input = tf.tidy(() => {
    const img = tf.browser.fromPixels(source);

    // padding image to square => [n, m] to [n, n], n > m
    const [h, w] = img.shape.slice(0, 2); // get source width and height
    const maxSize = Math.max(w, h); // get max size
    const imgPadded = img.pad([
      [0, maxSize - h], // padding y [bottom only]
      [0, maxSize - w], // padding x [right only]
      [0, 0],
    ]);

    xRatio = maxSize / w; // update xRatio
    yRatio = maxSize / h; // update yRatio

    return tf.image
      .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame
      .div(255.0) // normalize
      .expandDims(0); // add batch
  });

  return [input, xRatio, yRatio];
};

export const detect = async (source, model, canvasRef, callback = () => { }) => {
  const [modelWidth, modelHeight] = model.inputShape.slice(1, 3); // get model width and height

  tf.engine().startScope(); // start scoping tf engine
  const [input, xRatio, yRatio] = preprocess(source, modelWidth, modelHeight); // preprocess image

  const res = model.net.execute(input); // inference model
  const transRes = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
  const boxes = tf.tidy(() => {
    const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
    const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
    const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
    const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
    return tf
      .concat(
        [
          y1,
          x1,
          tf.add(y1, h), // y2
          tf.add(x1, w), // x2
        ],
        2
      )
      .squeeze();
  }); // process boxes [y1, x1, y2, x2]

  const [scores, classes] = tf.tidy(() => {
    // class scores
    const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); // #6 only squeeze axis 0 to handle only 1 class models
    return [rawScores.max(1), rawScores.argMax(1)];
  }); // get max scores and classes index

  const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // NMS to filter boxes

  const boxes_data = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
  const scores_data = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
  const classes_data = classes.gather(nms, 0).dataSync(); // indexing classes by nms index

  // Transform boxes to 4-point polygons
  const polygons_data = [];
  for (let i = 0; i < boxes_data.length; i += 4) {
    const y1 = boxes_data[i];
    const x1 = boxes_data[i + 1];
    const y2 = boxes_data[i + 2];
    const x2 = boxes_data[i + 3];

    // Define the 4 points of the polygon (rectangle in this case)
    polygons_data.push(
      y1, x1, // top-left
      y1, x2, // top-right
      y2, x2, // bottom-right
      y2, x1  // bottom-left
    );
  }

  renderPolygons(canvasRef, polygons_data, scores_data, classes_data, [xRatio, yRatio]); // render polygons
  tf.dispose([res, transRes, boxes, scores, classes, nms]); // clear memory

  callback();

  tf.engine().endScope(); // end of scoping
};


/**
 * Function run inference and do detection from source.
 * @param {HTMLImageElement|HTMLVideoElement} source
 * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
 */
export const detect_obb = async (source, model, canvasRef, callback = () => { }) => {
  tf.engine().startScope(); // start scoping tf engine
  const [input, xRatio, yRatio] = preprocess(source, 640, 640); // preprocess image

  const res = model.net.execute(input); // inference model
  const transRes = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]

  const boxes = tf.tidy(() => {
    const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
    const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
    const x_center = transRes.slice([0, 0, 0], [-1, -1, 1]); // x center
    const y_center = transRes.slice([0, 0, 1], [-1, -1, 1]); // y center
    const rotation = transRes.slice([0, 0, transRes.shape[2] - 1], [-1, -1, 1]); // rotation, between -π/2 to π/2 radians
    const boxes = tf.concat([x_center, y_center, w, h, rotation], 2).squeeze(); // x_center, y_center, width, height, rotation
    return boxes;
  });

  const [scores, classes] = tf.tidy(() => {
    const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); // class scores
    return [rawScores.max(1), rawScores.argMax(1)];
  });

  const boxesForNMS = boxes.slice([0, 0], [-1, 4]);

  const nms = await tf.image.nonMaxSuppressionAsync(boxesForNMS, scores, 500, 0.45, 0.2); // NMS to filter boxes

  const boxes_data = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
  const scores_data = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
  const classes_data = classes.gather(nms, 0).dataSync(); // indexing classes by nms index

  // Convert boxes and rotation to 4 corners
  const polygons_data = [];
  for (let i = 0; i < boxes_data.length; i += 5) {
    const x_center = boxes_data[i] * xRatio;
    const y_center = boxes_data[i + 1] * yRatio;
    const width = boxes_data[i + 2] * xRatio;
    const height = boxes_data[i + 3] * yRatio;
    const radians = boxes_data[i + 4];

    const corners = getCorners(x_center, y_center, width, height, radians);

    for (const [x, y] of corners) {
      polygons_data.push(y, x); // Notice the y, x order for correct format
    }
  }

  renderPolygons(canvasRef, polygons_data, scores_data, classes_data, [xRatio, yRatio]); // render polygons

  tf.dispose([res, transRes, boxes, scores, classes, nms]); // clear memory
  tf.engine().endScope(); // end of scoping
  return [boxes_data, scores_data, classes_data];
};

// Function to calculate the 4 corners of the rotated bounding box
function getCorners(x_center, y_center, width, height, radians) {
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);

  const halfWidth = width / 2;
  const halfHeight = height / 2;

  const corners = [
    [-halfWidth, -halfHeight],
    [halfWidth, -halfHeight],
    [halfWidth, halfHeight],
    [-halfWidth, halfHeight]
  ];

  return corners.map(([dx, dy]) => {
    const x = x_center + dx * cos - dy * sin;
    const y = y_center + dx * sin + dy * cos;
    return [x, y];
  });
}