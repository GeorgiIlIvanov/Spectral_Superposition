import React, { useState, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import Level0_Architecture from "./components/Level0_Architecture";
import Level1_ThomsonSphere from "./components/Level1_ThomsonSphere";
import Level2_InteractiveImage from "./components/Level2_InteractiveImage";
import { generateFeatureMatrix } from "./utils/superposition";
import "./App.css";

function App() {
  // State management
  const [selectedConfig, setSelectedConfig] = useState(6); // Default: Octahedron
  const [highlightedFeature, setHighlightedFeature] = useState(null);
  const [hiddenDim] = useState(2); // Bottleneck dimension

  // Trajectory geometry sequence from Level 2
  const [trajectoryGeometries, setTrajectoryGeometries] = useState(null);

  // Playback control reference from Level 2
  const [playbackControl, setPlaybackControl] = useState(null);

  // Toggle states for each level
  const [level0Open, setLevel0Open] = useState(true);
  const [level1Open, setLevel1Open] = useState(true);
  const [level2Open, setLevel2Open] = useState(true);

  // Calculate dynamic heights based on open/closed states
  const calculateHeights = useMemo(() => {
    const collapsedHeight = 50; // Height of collapsed section in px
    const openLevels = [level0Open, level1Open, level2Open].filter(Boolean).length;

    if (openLevels === 0) {
      return { level0: collapsedHeight, level1: collapsedHeight, level2: collapsedHeight };
    }

    // Calculate available viewport height after accounting for collapsed sections
    const totalCollapsed = (3 - openLevels) * collapsedHeight;
    const availableHeight = `calc((100vh - ${totalCollapsed}px) / ${openLevels})`;

    return {
      level0: level0Open ? availableHeight : `${collapsedHeight}px`,
      level1: level1Open ? availableHeight : `${collapsedHeight}px`,
      level2: level2Open ? availableHeight : `${collapsedHeight}px`
    };
  }, [level0Open, level1Open, level2Open]);

  // Generate feature matrix W based on selected configuration
  const W = useMemo(() => {
    return generateFeatureMatrix(selectedConfig, hiddenDim);
  }, [selectedConfig, hiddenDim]);

  // Handle configuration selection from Level 2
  const handleConfigSelect = (numFeatures) => {
    setSelectedConfig(numFeatures);
    setHighlightedFeature(null); // Reset highlight when changing config
  };

  // Handle trajectory geometry sequence from Level 2
  const handleTrajectoryUpdate = (trajectoryData) => {
    setTrajectoryGeometries(trajectoryData);

    // Set base configuration to the starting geometry
    if (trajectoryData && trajectoryData.length > 0 && trajectoryData[0].isStart) {
      const baseGeometry = trajectoryData[0].geometry;
      setSelectedConfig(baseGeometry.numFeatures);
    }
  };

  // Handle feature click from Level 1
  const handleFeatureClick = (featureIndex) => {
    setHighlightedFeature(featureIndex);
  };

  return (
    <div className="app-container">
      {/* Info panel */}
      <div className="info-panel">
        <div className="info-item">
          <span className="label">Features (n):</span>
          <span className="value">{selectedConfig}</span>
        </div>
        <div className="info-item">
          <span className="label">Hidden Dim (m):</span>
          <span className="value">{hiddenDim}</span>
        </div>
        {highlightedFeature !== null && (
          <div className="info-item highlighted">
            <span className="label">Selected Feature:</span>
            <span className="value">{highlightedFeature}</span>
          </div>
        )}
      </div>

      {/* Vertical three-level layout */}
      <div className="vertical-container">
        {/* Top Third: Level 2 - Feature Dimensionality */}
        <div
          className={`level-section top ${!level2Open ? 'collapsed' : ''}`}
          style={{ height: calculateHeights.level2 }}
        >
          <div
            className="level-label clickable"
            onClick={() => setLevel2Open(!level2Open)}
          >
            Level 2: Feature Geometry
          </div>
          {level2Open && (
            <Level2_InteractiveImage
              onConfigSelect={handleConfigSelect}
              selectedConfig={selectedConfig}
              onTrajectoryUpdate={handleTrajectoryUpdate}
              onPlaybackControl={setPlaybackControl}
            />
          )}
        </div>

        {/* Middle Third: Level 1 - Thomson Sphere */}
        <div
          className={`level-section middle ${!level1Open ? 'collapsed' : ''}`}
          style={{ height: calculateHeights.level1 }}
        >
          <div
            className="level-label clickable"
            onClick={() => setLevel1Open(!level1Open)}
          >
            Level 1: Concept Sphere
          </div>
          {level1Open && (
            <Canvas className="canvas">
              <PerspectiveCamera makeDefault position={[0, 0, 8]} />
              <OrbitControls enableDamping dampingFactor={0.05} />
              <ambientLight intensity={0.5} />
              <directionalLight position={[10, 10, 5]} intensity={1} />
              <directionalLight position={[-10, -10, -5]} intensity={0.5} />
              <pointLight position={[0, 0, 0]} intensity={0.3} />

              <Level1_ThomsonSphere
                numFeatures={selectedConfig}
                onFeatureClick={handleFeatureClick}
                highlightedFeature={highlightedFeature}
                trajectoryGeometries={trajectoryGeometries}
                playbackControl={playbackControl}
              />
            </Canvas>
          )}
        </div>

        {/* Bottom Third: Level 0 - Architecture */}
        <div
          className={`level-section bottom ${!level0Open ? 'collapsed' : ''}`}
          style={{ height: calculateHeights.level0 }}
        >
          <div
            className="level-label clickable"
            onClick={() => setLevel0Open(!level0Open)}
          >
            Level 0: Model Architecture
          </div>
          {level0Open && (
            <Canvas className="canvas">
              <PerspectiveCamera makeDefault position={[0, 0, 10]} />
              <OrbitControls enableDamping dampingFactor={0.05} />
              <ambientLight intensity={0.5} />
              <directionalLight position={[10, 10, 5]} intensity={1} />
              <directionalLight position={[-10, -10, -5]} intensity={0.5} />
              <pointLight position={[0, 0, 0]} intensity={0.3} />

              <Level0_Architecture
                W={W}
                highlightedFeature={highlightedFeature}
                inputDim={selectedConfig}
                hiddenDim={hiddenDim}
              />
            </Canvas>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
