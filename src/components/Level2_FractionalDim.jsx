import React, { useState } from "react";
import { getGeometricConfigurations } from "../utils/superposition";

/**
 * Level 2: Feature Fractional Dimensionality Diagram
 * Shows different geometric configurations and their fractional dimensionalities
 * Clicking on a configuration updates Level 1
 */
export default function Level2_FractionalDim({
  onConfigSelect,
  selectedConfig,
}) {
  const [hoveredConfig, setHoveredConfig] = useState(null);
  const configs = getGeometricConfigurations();

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "rgba(0, 0, 0, 0.6)",
        padding: "20px",
        color: "white",
        fontFamily: "monospace",
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <h3
        style={{
          margin: "0 0 15px 0",
          fontSize: "16px",
          fontWeight: "bold",
          borderBottom: "2px solid #F4BB64",
          paddingBottom: "8px",
        }}
      >
        Feature Dimensionality
      </h3>

      {/* Configuration grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
          gap: "12px",
          flex: 1,
        }}
      >
        {configs.map((config, i) => {
          const isSelected = selectedConfig === config.numFeatures;
          const isHovered = hoveredConfig === i;

          return (
            <div
              key={config.name}
              onClick={() =>
                onConfigSelect && onConfigSelect(config.numFeatures)
              }
              onMouseEnter={() => setHoveredConfig(i)}
              onMouseLeave={() => setHoveredConfig(null)}
              style={{
                padding: "12px",
                marginBottom: "8px",
                borderRadius: "8px",
                cursor: "pointer",
                background: isSelected
                  ? "rgba(78, 205, 196, 0.3)"
                  : isHovered
                  ? "rgba(255, 255, 255, 0.1)"
                  : "rgba(255, 255, 255, 0.05)",
                border: isSelected
                  ? `2px solid ${config.color}`
                  : "2px solid transparent",
                transition: "all 0.2s ease",
                transform: isHovered ? "translateX(5px)" : "translateX(0)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: "6px",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                  }}
                >
                  <div
                    style={{
                      width: "16px",
                      height: "16px",
                      borderRadius: "50%",
                      background: config.color,
                      boxShadow: `0 0 10px ${config.color}80`,
                    }}
                  />
                  <span
                    style={{
                      fontSize: "14px",
                      fontWeight: isSelected ? "bold" : "normal",
                    }}
                  >
                    {config.name}
                  </span>
                </div>
                <span
                  style={{
                    fontSize: "12px",
                    opacity: 0.7,
                    background: "rgba(255, 255, 255, 0.1)",
                    padding: "2px 8px",
                    borderRadius: "4px",
                  }}
                >
                  n={config.numFeatures}
                </span>
              </div>

              {/* Fractional dimensionality bar */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  fontSize: "11px",
                }}
              >
                <span style={{ opacity: 0.6, minWidth: "80px" }}>
                  Frac. Dim:
                </span>
                <div
                  style={{
                    flex: 1,
                    height: "6px",
                    background: "rgba(255, 255, 255, 0.1)",
                    borderRadius: "3px",
                    overflow: "hidden",
                    position: "relative",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${config.fractionalDim * 100}%`,
                      background: `linear-gradient(90deg, ${config.color}, ${config.color}CC)`,
                      transition: "width 0.3s ease",
                    }}
                  />
                </div>
                <span
                  style={{
                    fontWeight: "bold",
                    minWidth: "40px",
                    textAlign: "right",
                  }}
                >
                  {config.fractionalDim.toFixed(3)}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer with info and formula */}
      <div
        style={{
          marginTop: "15px",
          display: "flex",
          gap: "15px",
          alignItems: "center",
        }}
      >
        {/* Mathematical formula */}
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
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            minWidth: "200px",
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
