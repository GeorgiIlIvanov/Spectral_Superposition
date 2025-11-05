import React, { useState, useMemo, useRef, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { getThomsonPositions, getPolytopeEdges, getGeometricConfigurations } from '../utils/superposition';

// FEATURE FLAG: Set to false to disable trajectory drawing feature (mini-sphere)
const ENABLE_TRAJECTORY_DRAWING = false;

/**
 * Level 1: Thomson Problem Sphere Visualization
 * Shows polysemantic features as vectors on a sphere (W^T W interpretation)
 */
export default function Level1_ThomsonSphere({
  numFeatures = 6,
  onFeatureClick,
  highlightedFeature = null,
  trajectoryGeometries = null,
  playbackControl = null
}) {
  const [hoveredFeature, setHoveredFeature] = useState(null);
  const { camera } = useThree();

  // Rebasing state - tracks when mini-sphere becomes the new center
  const [isRebased, setIsRebased] = useState(false);
  const [originalBaseConfig, setOriginalBaseConfig] = useState(null);

  // Animation state
  const animationRef = useRef({
    isAnimating: false,
    progress: 0,
    startPosition: new THREE.Vector3(),
    targetPosition: new THREE.Vector3(),
    duration: 3000 // 3 seconds
  });

  // Get Thomson problem positions for the given number of features
  const featurePositions = useMemo(() => {
    return getThomsonPositions(numFeatures);
  }, [numFeatures]);

  // Get polytope edges
  const polytopeEdges = useMemo(() => {
    return getPolytopeEdges(numFeatures);
  }, [numFeatures]);

  // Get the configuration color for this geometry
  const configColor = useMemo(() => {
    const configs = getGeometricConfigurations();
    const config = configs.find(c => c.numFeatures === numFeatures);
    return config ? config.color : '#FFFFFF';
  }, [numFeatures]);

  // Process trajectory geometries for mini-sphere (only one - the end point)
  const miniSphere = useMemo(() => {
    if (!ENABLE_TRAJECTORY_DRAWING) return null; // DISABLED: Trajectory drawing feature

    if (!trajectoryGeometries || trajectoryGeometries.length < 2) {
      return null;
    }

    // Get the end geometry (second element)
    const endGeomInfo = trajectoryGeometries[1];
    if (!endGeomInfo || endGeomInfo.isStart) {
      return null;
    }

    const { geometry } = endGeomInfo;

    // Attach to the first vertex (index 0)
    const attachmentIndex = 0;
    const attachmentPos = featurePositions[attachmentIndex];

    if (!attachmentPos) {
      return null;
    }

    return {
      geometry,
      attachmentIndex,
      attachmentPos
    };
  }, [trajectoryGeometries, featurePositions]);

  // Reset rebasing when trajectory is cleared (new click)
  useEffect(() => {
    if (!ENABLE_TRAJECTORY_DRAWING) return; // DISABLED: Trajectory drawing feature

    if (!trajectoryGeometries) {
      console.log('Trajectory cleared - resetting to single base sphere');
      setIsRebased(false);
      setOriginalBaseConfig(null);

      // Reset camera to original position
      camera.position.set(0, 0, 8);
      camera.lookAt(0, 0, 0);
    }
  }, [trajectoryGeometries, camera]);

  // Detect when playback starts and trigger camera animation
  useEffect(() => {
    if (!ENABLE_TRAJECTORY_DRAWING) return; // DISABLED: Trajectory drawing feature

    if (!trajectoryGeometries || trajectoryGeometries.length < 2 || !miniSphere) {
      return;
    }

    const endGeomInfo = trajectoryGeometries[1];
    if (endGeomInfo && endGeomInfo.playbackStarted && !animationRef.current.isAnimating) {
      // Store original base config before rebasing
      setOriginalBaseConfig({
        numFeatures,
        featurePositions: [...featurePositions],
        polytopeEdges: [...polytopeEdges],
        configColor
      });

      // Start camera animation to mini-sphere
      animationRef.current.isAnimating = true;
      animationRef.current.progress = 0;
      animationRef.current.startPosition.copy(camera.position);
      animationRef.current.startTime = Date.now();

      // Calculate target position: zoom closer to the mini-sphere
      const miniSphereWorldPos = new THREE.Vector3(
        miniSphere.attachmentPos.x * 2,
        miniSphere.attachmentPos.y * 2,
        miniSphere.attachmentPos.z * 2
      );

      // Position camera at a reasonable distance from the mini-sphere
      const offset = miniSphereWorldPos.clone().normalize().multiplyScalar(1.5);
      animationRef.current.targetPosition.copy(miniSphereWorldPos.clone().add(offset));

      console.log('Starting camera zoom animation to mini-sphere');
    }
  }, [trajectoryGeometries, miniSphere, camera, numFeatures, featurePositions, polytopeEdges, configColor]);

  // Animation loop
  useFrame(() => {
    if (!animationRef.current.isAnimating) return;

    const elapsed = Date.now() - animationRef.current.startTime;
    const progress = Math.min(elapsed / animationRef.current.duration, 1);

    // Easing function (ease-in-out)
    const eased = progress < 0.5
      ? 2 * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 2) / 2;

    // Interpolate camera position
    camera.position.lerpVectors(
      animationRef.current.startPosition,
      animationRef.current.targetPosition,
      eased
    );

    // Make camera look at the mini-sphere center
    if (miniSphere) {
      const lookAtPos = new THREE.Vector3(
        miniSphere.attachmentPos.x * 2,
        miniSphere.attachmentPos.y * 2,
        miniSphere.attachmentPos.z * 2
      );
      camera.lookAt(lookAtPos);
    }

    // Animation complete
    if (progress >= 1) {
      animationRef.current.isAnimating = false;
      console.log('Camera zoom animation complete - rebasing to mini-sphere');

      // Rebase: mini-sphere becomes the new center
      setIsRebased(true);

      // Stop video playback
      if (playbackControl && playbackControl.stopPlayback) {
        playbackControl.stopPlayback();
      }
    }
  });

  // Colors for features
  const colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
    '#F8B739', '#52B788'
  ];

  // If rebased, show mini-sphere as new center and original base sphere offset
  if (ENABLE_TRAJECTORY_DRAWING && isRebased && miniSphere && originalBaseConfig) {
    const miniSphereGeometry = miniSphere.geometry;
    const newCenterPositions = getThomsonPositions(miniSphereGeometry.numFeatures);
    const newCenterEdges = getPolytopeEdges(miniSphereGeometry.numFeatures);
    const newCenterColor = (() => {
      const configs = getGeometricConfigurations();
      const config = configs.find(c => c.numFeatures === miniSphereGeometry.numFeatures);
      return config ? config.color : '#FFFFFF';
    })();

    // Calculate offset for old base sphere (move it away from center)
    const oldSphereOffset = new THREE.Vector3(-4, 2, -2);

    return (
      <group>
        {/* New center sphere (was mini-sphere) - now at origin with radius 2 */}
        <mesh>
          <sphereGeometry args={[2, 64, 64]} />
          <meshBasicMaterial
            color={newCenterColor}
            transparent
            opacity={0.1}
            wireframe
          />
        </mesh>

        {/* New center feature vectors */}
        {newCenterPositions.map((pos, i) => {
          const scaledPos = {
            x: pos.x * 2,
            y: pos.y * 2,
            z: pos.z * 2
          };

          return (
            <Arrow
              key={`new-center-feature-${i}`}
              start={[0, 0, 0]}
              end={[scaledPos.x, scaledPos.y, scaledPos.z]}
              color={colors[i % colors.length]}
              highlighted={false}
            />
          );
        })}

        {/* New center polytope edges */}
        {newCenterEdges.map(([i, j], edgeIndex) => {
          const pos1 = newCenterPositions[i];
          const pos2 = newCenterPositions[j];

          if (!pos1 || !pos2) return null;

          const points = [
            new THREE.Vector3(pos1.x * 2, pos1.y * 2, pos1.z * 2),
            new THREE.Vector3(pos2.x * 2, pos2.y * 2, pos2.z * 2)
          ];
          const geometry = new THREE.BufferGeometry().setFromPoints(points);

          return (
            <line key={`new-center-edge-${edgeIndex}`} geometry={geometry}>
              <lineBasicMaterial
                color={newCenterColor}
                opacity={0.6}
                transparent
                linewidth={2}
              />
            </line>
          );
        })}

        {/* Original base sphere - offset and dimmed */}
        <group position={[oldSphereOffset.x, oldSphereOffset.y, oldSphereOffset.z]}>
          <mesh>
            <sphereGeometry args={[0.6, 32, 32]} />
            <meshBasicMaterial
              color={originalBaseConfig.configColor}
              transparent
              opacity={0.05}
              wireframe
            />
          </mesh>

          {/* Original base feature vectors - smaller and dimmed */}
          {originalBaseConfig.featurePositions.map((pos, i) => {
            const scaledPos = {
              x: pos.x * 0.6,
              y: pos.y * 0.6,
              z: pos.z * 0.6
            };

            return (
              <Arrow
                key={`old-base-feature-${i}`}
                start={[0, 0, 0]}
                end={[scaledPos.x, scaledPos.y, scaledPos.z]}
                color={colors[i % colors.length]}
                highlighted={false}
              />
            );
          })}

          {/* Original base polytope edges */}
          {originalBaseConfig.polytopeEdges.map(([i, j], edgeIndex) => {
            const pos1 = originalBaseConfig.featurePositions[i];
            const pos2 = originalBaseConfig.featurePositions[j];

            if (!pos1 || !pos2) return null;

            const points = [
              new THREE.Vector3(pos1.x * 0.6, pos1.y * 0.6, pos1.z * 0.6),
              new THREE.Vector3(pos2.x * 0.6, pos2.y * 0.6, pos2.z * 0.6)
            ];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);

            return (
              <line key={`old-base-edge-${edgeIndex}`} geometry={geometry}>
                <lineBasicMaterial
                  color={originalBaseConfig.configColor}
                  opacity={0.3}
                  transparent
                  linewidth={1}
                />
              </line>
            );
          })}
        </group>
      </group>
    );
  }

  // Normal rendering (not rebased)
  return (
    <group>
      {/* Transparent sphere to show the unit sphere */}
      <mesh>
        <sphereGeometry args={[2, 64, 64]} />
        <meshBasicMaterial
          color="#FFFFFF"
          transparent
          opacity={0.05}
          wireframe
        />
      </mesh>

      {/* Feature vectors */}
      {featurePositions.map((pos, i) => {
        const isHighlighted = highlightedFeature === i || hoveredFeature === i;
        const scale = isHighlighted ? 1.3 : 1.0;

        // Scale position for sphere radius of 2
        const scaledPos = {
          x: pos.x * 2,
          y: pos.y * 2,
          z: pos.z * 2
        };

        return (
          <group
            key={`feature-${i}`}
            onClick={() => onFeatureClick && onFeatureClick(i)}
            onPointerOver={() => setHoveredFeature(i)}
            onPointerOut={() => setHoveredFeature(null)}
          >
            {/* Feature vector as arrow */}
            <Arrow
              start={[0, 0, 0]}
              end={[scaledPos.x, scaledPos.y, scaledPos.z]}
              color={colors[i % colors.length]}
              highlighted={isHighlighted}
            />
          </group>
        );
      })}

      {/* Polytope edges connecting vertices */}
      {polytopeEdges.map(([i, j], edgeIndex) => {
        const pos1 = featurePositions[i];
        const pos2 = featurePositions[j];

        if (!pos1 || !pos2) return null;

        const points = [
          new THREE.Vector3(pos1.x * 2, pos1.y * 2, pos1.z * 2),
          new THREE.Vector3(pos2.x * 2, pos2.y * 2, pos2.z * 2)
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        return (
          <line key={`edge-${edgeIndex}`} geometry={geometry}>
            <lineBasicMaterial
              color={configColor}
              opacity={0.6}
              transparent
              linewidth={2}
            />
          </line>
        );
      })}

      {/* Mini-sphere for end point geometry (only shown when not rebased) */}
      {!isRebased && miniSphere && miniSphere.attachmentPos && (
        <MiniSphere
          position={{
            x: miniSphere.attachmentPos.x * 2,
            y: miniSphere.attachmentPos.y * 2,
            z: miniSphere.attachmentPos.z * 2
          }}
          geometry={miniSphere.geometry}
          colors={colors}
        />
      )}
    </group>
  );
}

// Arrow component for feature vectors
function Arrow({ start, end, color, highlighted }) {
  const direction = new THREE.Vector3(
    end[0] - start[0],
    end[1] - start[1],
    end[2] - start[2]
  );
  const length = direction.length();
  direction.normalize();

  const arrowHelper = useMemo(() => {
    const helper = new THREE.ArrowHelper(
      direction,
      new THREE.Vector3(...start),
      length,
      color,
      length * 0.15,
      length * 0.1
    );
    return helper;
  }, [start, end, color, length]);

  // Update arrow opacity based on highlight
  if (arrowHelper.line && arrowHelper.cone) {
    arrowHelper.line.material.opacity = highlighted ? 0.8 : 0.4;
    arrowHelper.line.material.transparent = true;
    arrowHelper.cone.material.opacity = highlighted ? 0.9 : 0.5;
    arrowHelper.cone.material.transparent = true;
  }

  return <primitive object={arrowHelper} />;
}

// MiniSphere component for trajectory geometries
function MiniSphere({ position, geometry, colors }) {
  const miniRadius = 0.6; // 3x larger than original (0.2 * 3)

  // Get Thomson positions for this mini-sphere's geometry
  const miniFeaturePositions = useMemo(() => {
    return getThomsonPositions(geometry.numFeatures);
  }, [geometry.numFeatures]);

  // Get polytope edges for this geometry
  const miniPolytopeEdges = useMemo(() => {
    return getPolytopeEdges(geometry.numFeatures);
  }, [geometry.numFeatures]);

  // Get the configuration color for this geometry
  const miniConfigColor = useMemo(() => {
    const configs = getGeometricConfigurations();
    const config = configs.find(c => c.numFeatures === geometry.numFeatures);
    return config ? config.color : '#FFFFFF';
  }, [geometry.numFeatures]);

  return (
    <group position={[position.x, position.y, position.z]}>
      {/* Mini transparent sphere */}
      <mesh>
        <sphereGeometry args={[miniRadius, 32, 32]} />
        <meshBasicMaterial
          color={miniConfigColor}
          transparent
          opacity={0.15}
          wireframe
        />
      </mesh>

      {/* Mini feature vectors */}
      {miniFeaturePositions.map((pos, i) => {
        const scaledPos = {
          x: pos.x * miniRadius,
          y: pos.y * miniRadius,
          z: pos.z * miniRadius
        };

        return (
          <Arrow
            key={`mini-feature-${i}`}
            start={[0, 0, 0]}
            end={[scaledPos.x, scaledPos.y, scaledPos.z]}
            color={colors[i % colors.length]}
            highlighted={false}
          />
        );
      })}

      {/* Mini polytope edges */}
      {miniPolytopeEdges.map(([i, j], edgeIndex) => {
        const pos1 = miniFeaturePositions[i];
        const pos2 = miniFeaturePositions[j];

        if (!pos1 || !pos2) return null;

        const points = [
          new THREE.Vector3(pos1.x * miniRadius, pos1.y * miniRadius, pos1.z * miniRadius),
          new THREE.Vector3(pos2.x * miniRadius, pos2.y * miniRadius, pos2.z * miniRadius)
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        return (
          <line key={`mini-edge-${edgeIndex}`} geometry={geometry}>
            <lineBasicMaterial
              color={miniConfigColor}
              opacity={0.5}
              transparent
              linewidth={1}
            />
          </line>
        );
      })}
    </group>
  );
}
