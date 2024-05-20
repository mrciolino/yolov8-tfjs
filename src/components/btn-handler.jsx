import { useState, useRef } from "react";

const ButtonHandler = ({ imageRef, cameraRef, videoRef }) => {
  const [streaming, setStreaming] = useState(null); // streaming state
  const inputImageRef = useRef(null); // video input reference

  // closing image
  const closeImage = () => {
    const url = imageRef.current.src;
    imageRef.current.src = "#"; // restore image source
    URL.revokeObjectURL(url); // revoke url

    setStreaming(null); // set streaming to null
    inputImageRef.current.value = ""; // reset input image
    imageRef.current.style.display = "none"; // hide image
  };

  return (
    <div className="btn-container">
      {/* Image Handler */}
      <input
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          const url = URL.createObjectURL(e.target.files[0]); // create blob url
          imageRef.current.src = url; // set video source
          imageRef.current.style.display = "block"; // show video
          setStreaming("image"); // set streaming to video
        }}
        ref={inputImageRef}
      />
      <button
        onClick={() => {
          // if not streaming
          if (streaming === null) inputImageRef.current.click();
          // closing image streaming
          else if (streaming === "image") closeImage();
          else alert(`Can't handle more than 1 stream\nCurrently streaming : ${streaming}`); // if streaming video or webcam
        }}
      >
        {streaming === "image" ? "Close" : "Open"} Image
      </button>
    </div>
  );
};

export default ButtonHandler;
