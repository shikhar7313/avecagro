// components/LoadingBar.js
import React, { useState, useEffect } from "react";

const LoadingBar = ({ duration }) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const interval = 50;
    const increment = 100 / (duration / interval);

    const timer = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(timer);
          return 100;
        }
        return prev + increment;
      });
    }, interval);

    return () => clearInterval(timer);
  }, [duration]);

  return (
    <div style={{ textAlign: "center" }}>
      {/* Glowing Progress Bar */}
      <div
        style={{
          position: "relative",
          width: "80%",
          height: "20px",
          margin: "0 auto",
          border: "2px solid #00ff00",
          borderRadius: "8px",
          background: "rgba(0, 0, 0, 0.6)",
          boxShadow: "0 0 10px #00ff00, 0 0 20px #00ff00 inset",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${progress}%`,
            background: "linear-gradient(90deg, #00ff00, #33ff33)",
            borderRadius: "6px",
            boxShadow: "0 0 15px #00ff00, 0 0 30px #00ff00",
            transition: "width 0.05s linear",
          }}
        />
      </div>

      {/* Percentage */}
      <p
        style={{
          marginTop: "10px",
          fontFamily: "monospace",
          color: "#00ff00",
          fontSize: "18px",
          textShadow: "0 0 8px #00ff00",
        }}
      >
        {Math.floor(progress)}%
      </p>
    </div>
  );
};

export default LoadingBar;
