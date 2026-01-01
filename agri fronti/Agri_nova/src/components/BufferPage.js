// components/BufferPage.js
import React from "react";
import LoadingBar from "./loadingbar.js";

const BufferPage = ({ videoSrc, duration }) => {
  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        overflow: "hidden",
        zIndex: 9999,
        backgroundColor: "black",
      }}
    >
      {/* Background Video */}
      <video
        src={videoSrc}
        autoPlay
        muted
        loop
        playsInline
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          minWidth: "100%",
          minHeight: "100%",
          objectFit: "cover",
          zIndex: 1,
        }}
      />

      {/* Bottom Loading Bar */}
      <div
        style={{
          position: "absolute",
          bottom: "25px", // distance from bottom
          left: "50%",
          transform: "translateX(-50%)",
          width: "100%",
          zIndex: 2,
        }}
      >
        <LoadingBar duration={duration} />
      </div>
    </div>
  );
};

export default BufferPage;
