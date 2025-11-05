import React, { useState, useRef, useEffect } from "react";
import { rgbToValue, fractionalDimToGeometry } from "../utils/colorbarMapping";

// FEATURE FLAG: Set to false to disable trajectory drawing feature
const ENABLE_TRAJECTORY_DRAWING = false;

// MEDIA TOGGLE: Set to true to use video instead of static image
const USE_VIDEO = false;

/**
 * Level 2: Interactive Feature Geometry Video
 * Scrub through video timeline and click to select geometry based on fractional dimensionality
 */
export default function Level2_InteractiveImage({
  onConfigSelect,
  selectedConfig,
  onTrajectoryUpdate,
  onPlaybackControl,
}) {
  const [hoverInfo, setHoverInfo] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const animationFrameRef = useRef(null);

  // Trajectory drawing state
  const [isDrawing, setIsDrawing] = useState(false);
  const [trajectoryPoints, setTrajectoryPoints] = useState([]);
  const trajectoryCanvasRef = useRef(null);

  // Track start and end geometry only
  const [startGeometry, setStartGeometry] = useState(null);
  const [endGeometry, setEndGeometry] = useState(null);

  // Expose playback control to parent
  useEffect(() => {
    if (onPlaybackControl) {
      onPlaybackControl({
        stopPlayback: () => {
          const video = videoRef.current;
          if (video) {
            video.pause();
            setIsPlaying(false);
          }
        }
      });
    }
  }, [onPlaybackControl]);

  // Convert time to scale (logarithmic mapping from 10^5 to 10^-15)
  const timeToScale = (time) => {
    if (duration === 0) return { coefficient: 1, exponent: 5 };
    const progress = time / duration; // 0 to 1
    // Linear interpolation in log space: log(10^5) to log(10^-15)
    // log10(10^5) = 5, log10(10^-15) = -15
    const logScale = 5 - progress * 20; // 5 to -15
    const fullValue = Math.pow(10, logScale);

    // Split into coefficient and whole number exponent
    const exponent = Math.floor(logScale);
    const coefficient = fullValue / Math.pow(10, exponent);

    return { coefficient, exponent };
  };

  // Load and setup video or image
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    if (USE_VIDEO) {
      // VIDEO MODE
      const video = videoRef.current;

      const handleLoadedMetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        setDuration(video.duration);

        // Set trajectory canvas size to match
        if (trajectoryCanvasRef.current) {
          trajectoryCanvasRef.current.width = video.videoWidth;
          trajectoryCanvasRef.current.height = video.videoHeight;
        }

        // Set to first frame
        video.currentTime = 0;
      };

      const handleLoadedData = () => {
        // Draw first frame once video data is actually loaded
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        setVideoLoaded(true);
      };

      const handleSeeked = () => {
        // Draw current frame to canvas when seeking
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      };

      const handleTimeUpdate = () => {
        setCurrentTime(video.currentTime);
      };

      video.addEventListener("loadedmetadata", handleLoadedMetadata);
      video.addEventListener("loadeddata", handleLoadedData);
      video.addEventListener("seeked", handleSeeked);
      video.addEventListener("timeupdate", handleTimeUpdate);

      // Load the feature geometry video
      video.src = "/feature-geometry.mp4";
      video.load();

      return () => {
        video.removeEventListener("loadedmetadata", handleLoadedMetadata);
        video.removeEventListener("loadeddata", handleLoadedData);
        video.removeEventListener("seeked", handleSeeked);
        video.removeEventListener("timeupdate", handleTimeUpdate);
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      };
    } else {
      // IMAGE MODE
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;

        // Set trajectory canvas size to match
        if (trajectoryCanvasRef.current) {
          trajectoryCanvasRef.current.width = img.width;
          trajectoryCanvasRef.current.height = img.height;
        }

        // Draw the image
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        setVideoLoaded(true); // Reuse this flag for "content loaded"
      };
      img.src = "/feature_geometry.png";
    }
  }, []);

  // Update canvas during playback (video mode only)
  useEffect(() => {
    if (!USE_VIDEO || !isPlaying || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    const updateCanvas = () => {
      if (video.paused || video.ended) {
        setIsPlaying(false);
        return;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      animationFrameRef.current = requestAnimationFrame(updateCanvas);
    };

    updateCanvas();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isPlaying]);

  // Draw trajectory on overlay canvas
  useEffect(() => {
    if (!ENABLE_TRAJECTORY_DRAWING) return; // DISABLED: Trajectory drawing feature

    const trajectoryCanvas = trajectoryCanvasRef.current;
    if (!trajectoryCanvas || trajectoryPoints.length === 0) return;

    const ctx = trajectoryCanvas.getContext("2d");
    ctx.clearRect(0, 0, trajectoryCanvas.width, trajectoryCanvas.height);

    if (trajectoryPoints.length < 2) return;

    // Draw the trajectory line
    ctx.strokeStyle = "#B8860B"; // Rustic gold color
    ctx.lineWidth = 4; // Thicker than Level 1 lines, thinner than slider
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctx.beginPath();
    ctx.moveTo(trajectoryPoints[0].x, trajectoryPoints[0].y);

    for (let i = 1; i < trajectoryPoints.length; i++) {
      ctx.lineTo(trajectoryPoints[i].x, trajectoryPoints[i].y);
    }

    ctx.stroke();
  }, [trajectoryPoints]);

  // Toggle play/pause (video mode only)
  const togglePlayPause = () => {
    if (!USE_VIDEO) return;

    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
      setIsPlaying(false);
    } else {
      // Clear trajectory when starting playback
      if (ENABLE_TRAJECTORY_DRAWING) {
        setTrajectoryPoints([]);
      }

      video.play();
      setIsPlaying(true);

      // Notify parent that playback started (for camera animation)
      if (ENABLE_TRAJECTORY_DRAWING && onTrajectoryUpdate && startGeometry && endGeometry) {
        const trajectoryData = [
          { geometry: startGeometry, isStart: true },
          { geometry: endGeometry, isStart: false, playbackStarted: true }
        ];
        onTrajectoryUpdate(trajectoryData);
      }
    }
  };

  // Handle time slider change (video mode only)
  const handleTimeChange = (e) => {
    if (!USE_VIDEO) return;

    const newTime = parseFloat(e.target.value);
    setCurrentTime(newTime);
    if (videoRef.current) {
      videoRef.current.currentTime = newTime;
    }
  };

  const handleMouseDown = (e) => {
    if (!videoLoaded || !ENABLE_TRAJECTORY_DRAWING) return; // DISABLED: Trajectory drawing feature

    // Clear previous trajectory and geometries FIRST
    setTrajectoryPoints([]);
    setStartGeometry(null);
    setEndGeometry(null);

    // Clear on parent side by sending null
    if (onTrajectoryUpdate) {
      onTrajectoryUpdate(null);
    }

    setIsDrawing(true);

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    setTrajectoryPoints([{ x, y }]);

    // Get geometry at starting point
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const pixel = ctx.getImageData(x, y, 1, 1).data;
    const [r, g, b] = pixel;
    const fractionalDim = rgbToValue(r, g, b);
    const geometry = fractionalDimToGeometry(fractionalDim);

    console.log('Starting point - RGB:', [r, g, b], 'FracDim:', fractionalDim, 'Geometry:', geometry.name, 'numFeatures:', geometry.numFeatures);

    // Set start geometry only (end stays null until mouse up)
    setStartGeometry(geometry);
  };

  const handleMouseMove = (e) => {
    if (!videoLoaded) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    setMousePos({ x: e.clientX, y: e.clientY });

    // Get pixel color at mouse position
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const pixel = ctx.getImageData(x, y, 1, 1).data;
    const [r, g, b] = pixel;

    // Convert RGB to fractional dimensionality
    const fractionalDim = rgbToValue(r, g, b);
    const geometry = fractionalDimToGeometry(fractionalDim);

    // If drawing, add point to trajectory and update end geometry
    if (ENABLE_TRAJECTORY_DRAWING && isDrawing) {
      setTrajectoryPoints((prev) => [...prev, { x, y }]);
      // Continuously update the end geometry as we draw
      setEndGeometry(geometry);
    }

    setHoverInfo({
      fractionalDim,
      geometry,
      rgb: [r, g, b],
    });
  };

  const handleMouseUp = () => {
    if (!ENABLE_TRAJECTORY_DRAWING) return; // DISABLED: Trajectory drawing feature

    setIsDrawing(false);

    // Log end point and send trajectory to parent
    if (endGeometry) {
      console.log('Ending point - Geometry:', endGeometry.name, 'numFeatures:', endGeometry.numFeatures);

      // Send start and end geometry to parent
      if (onTrajectoryUpdate && startGeometry) {
        const trajectoryData = [
          { geometry: startGeometry, isStart: true },
          { geometry: endGeometry, isStart: false }
        ];
        onTrajectoryUpdate(trajectoryData);
      }
    }
  };

  const handleMouseLeave = () => {
    setHoverInfo(null);
    if (ENABLE_TRAJECTORY_DRAWING && isDrawing) {
      setIsDrawing(false);

      // Send trajectory even if mouse leaves
      if (endGeometry && onTrajectoryUpdate && startGeometry) {
        console.log('Ending point (mouse left) - Geometry:', endGeometry.name, 'numFeatures:', endGeometry.numFeatures);
        const trajectoryData = [
          { geometry: startGeometry, isStart: true },
          { geometry: endGeometry, isStart: false }
        ];
        onTrajectoryUpdate(trajectoryData);
      }
    }
  };

  const handleClick = (e) => {
    // Only select geometry if not drawing
    if (!hoverInfo || isDrawing) return;

    // Select the geometry
    onConfigSelect(hoverInfo.geometry.numFeatures);
  };

  // Get color for geometry name based on fractional dimensionality
  const getGeometryColor = (fractionalDim) => {
    if (fractionalDim < 0.35) return "#DC143C"; // Dark red (no feature)
    if (fractionalDim >= 0.36 && fractionalDim < 0.39) return "#C77DFF"; // Purple (square antiprism)
    if (fractionalDim >= 0.39 && fractionalDim < 0.48) return "#FFA07A"; // Orange (pentagon)
    if (fractionalDim >= 0.48 && fractionalDim < 0.62) return "#FFD700"; // Yellow (digon/octahedron)
    if (fractionalDim >= 0.62 && fractionalDim < 0.72) return "#52B788"; // Green (triangle)
    return "#4ECDC4"; // Blue (tetrahedron)
  };

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "rgba(0, 0, 0, 0.6)",
        padding: "20px",
        color: "white",
        fontFamily: "monospace",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Hidden video element */}
      <video ref={videoRef} style={{ display: "none" }} preload="auto" />

      {/* Interactive Canvas */}
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
          overflow: "hidden",
          minHeight: 0, // Allow flex child to shrink below content size
        }}
      >
        <div style={{
          position: "relative",
          display: "inline-block",
          width: "100%",
          height: "100%",
          maxWidth: "100%",
          maxHeight: "100%",
        }}>
          {/* Video canvas (background) */}
          <canvas
            ref={canvasRef}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "contain",
              border: "2px solid rgba(244, 187, 100, 0.3)",
              borderRadius: "8px",
              display: "block",
            }}
          />
          {/* Trajectory canvas (overlay) */}
          <canvas
            ref={trajectoryCanvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
            onClick={handleClick}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              cursor: isDrawing ? "crosshair" : "crosshair",
              pointerEvents: "all",
            }}
          />
        </div>

        {/* Hover Tooltip */}
        {hoverInfo && (
          <div
            style={{
              position: "fixed",
              left: mousePos.x + 15,
              top: mousePos.y + 15,
              background: "rgba(0, 0, 0, 0.9)",
              border: "2px solid #F4BB64",
              borderRadius: "8px",
              padding: "12px",
              fontSize: "12px",
              pointerEvents: "none",
              zIndex: 1000,
              minWidth: "200px",
              boxShadow: "0 4px 12px rgba(0, 0, 0, 0.5)",
            }}
          >
            <div
              style={{
                marginBottom: "8px",
                fontWeight: "bold",
                color: "#F4BB64",
              }}
            >
              Fractional Dimensionality
            </div>
            <div style={{ marginBottom: "8px" }}>
              <span style={{ opacity: 0.7 }}>Value:</span>{" "}
              <span style={{ fontWeight: "bold", color: "#FFD700" }}>
                {hoverInfo.fractionalDim.toFixed(3)}
              </span>
            </div>
            <div
              style={{
                padding: "8px",
                background: "rgba(255, 255, 255, 0.1)",
                borderRadius: "4px",
                marginTop: "8px",
              }}
            >
              <div
                style={{ fontSize: "11px", opacity: 0.7, marginBottom: "4px" }}
              >
                Geometry:
              </div>
              <div
                style={{
                  fontWeight: "bold",
                  color: getGeometryColor(hoverInfo.fractionalDim),
                }}
              >
                {hoverInfo.geometry.name}
              </div>
              <div style={{ fontSize: "11px", opacity: 0.7, marginTop: "4px" }}>
                n = {hoverInfo.geometry.numFeatures}
              </div>
            </div>
            <div
              style={{
                marginTop: "8px",
                fontSize: "10px",
                opacity: 0.5,
                fontStyle: "italic",
              }}
            >
              Click to select
            </div>
          </div>
        )}
      </div>

      {/* Playback Controls (only shown for video mode) */}
      {USE_VIDEO && (
        <div
          style={{
            marginTop: "15px",
            padding: "12px",
            background: "rgba(0, 0, 0, 0.4)",
            borderRadius: "8px",
          }}
        >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "12px",
          }}
        >
          {/* Play/Pause Button */}
          <button
            onClick={togglePlayPause}
            style={{
              width: "36px",
              height: "36px",
              borderRadius: "50%",
              background: isPlaying
                ? "rgba(180, 140, 70, 0.7)"
                : "rgba(180, 140, 70, 0.3)",
              border: "2px solid #B8860B",
              color: "#1a1a1a",
              fontSize: "14px",
              fontWeight: "bold",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "all 0.2s ease",
              outline: "none",
            }}
            onMouseEnter={(e) => {
              e.target.style.background = "rgba(180, 140, 70, 0.85)";
              e.target.style.transform = "scale(1.1)";
            }}
            onMouseLeave={(e) => {
              e.target.style.background = isPlaying
                ? "rgba(180, 140, 70, 0.7)"
                : "rgba(180, 140, 70, 0.3)";
              e.target.style.transform = "scale(1)";
            }}
          >
            {isPlaying ? "⏸" : "▶"}
          </button>

          {/* Scale Display */}
          <span style={{ fontSize: "11px", opacity: 0.7, minWidth: "100px" }}>
            {(() => {
              const { coefficient, exponent } = timeToScale(currentTime);
              return (
                <>
                  {coefficient.toFixed(2)} × 10<sup>{exponent}</sup>
                </>
              );
            })()}
          </span>

          {/* Custom Styled Slider with Progress Fill */}
          <div style={{ flex: 1, position: "relative", height: "8px" }}>
            {/* Background track */}
            <div
              style={{
                position: "absolute",
                width: "100%",
                height: "8px",
                background: "rgba(184, 134, 11, 0.15)",
                borderRadius: "4px",
              }}
            />
            {/* Progress fill */}
            <div
              style={{
                position: "absolute",
                width: `${(currentTime / duration) * 100}%`,
                height: "8px",
                background: "linear-gradient(90deg, #8B6914, #B8860B)",
                borderRadius: "4px",
                transition: "width 0.05s ease",
              }}
            />
            {/* Slider input */}
            <input
              type="range"
              min="0"
              max={duration}
              step="0.01"
              value={currentTime}
              onChange={handleTimeChange}
              style={{
                position: "absolute",
                width: "100%",
                height: "8px",
                background: "transparent",
                outline: "none",
                cursor: "pointer",
                WebkitAppearance: "none",
                appearance: "none",
              }}
            />
          </div>

          {/* End Scale Display */}
          <span
            style={{
              fontSize: "11px",
              opacity: 0.7,
              minWidth: "100px",
              textAlign: "right",
            }}
          >
            1.00 × 10<sup>-15</sup>
          </span>
        </div>
        </div>
      )}

      {/* Footer with formula */}
      <div
        style={{
          marginTop: "10px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            padding: "12px",
            background: "rgba(0, 0, 0, 0.4)",
            borderRadius: "8px",
            fontSize: "14px",
            textAlign: "center",
            fontFamily: "serif",
            fontStyle: "italic",
            color: "#FFD700",
          }}
        >
          <span>
            f(W, x) = ReLU(W<sup>T</sup>Wx + b)
          </span>
        </div>
      </div>
    </div>
  );
}
