import React, { useMemo } from 'react';
import * as THREE from 'three';

/**
 * Level 0: Neural Network Architecture Visualization
 * Shows the three layers: x -> h -> x' with connections via W and W^T
 */
export default function Level0_Architecture({
  W,
  highlightedFeature = null,
  inputDim = 5,
  hiddenDim = 2
}) {
  const layers = useMemo(() => {
    const layerSpacing = 3;

    // Input layer positions (x)
    const inputLayer = [];
    const inputSpacing = 0.8;
    const inputOffset = -(inputDim - 1) * inputSpacing / 2;
    for (let i = 0; i < inputDim; i++) {
      inputLayer.push({
        x: -layerSpacing,
        y: inputOffset + i * inputSpacing,
        z: 0,
        index: i
      });
    }

    // Hidden layer positions (h)
    const hiddenLayer = [];
    const hiddenSpacing = 1.5;
    const hiddenOffset = -(hiddenDim - 1) * hiddenSpacing / 2;
    for (let i = 0; i < hiddenDim; i++) {
      hiddenLayer.push({
        x: 0,
        y: hiddenOffset + i * hiddenSpacing,
        z: 0,
        index: i
      });
    }

    // Output layer positions (x')
    const outputLayer = [];
    const outputSpacing = 0.8;
    const outputOffset = -(inputDim - 1) * outputSpacing / 2;
    for (let i = 0; i < inputDim; i++) {
      outputLayer.push({
        x: layerSpacing,
        y: outputOffset + i * outputSpacing,
        z: 0,
        index: i
      });
    }

    return { inputLayer, hiddenLayer, outputLayer };
  }, [inputDim, hiddenDim]);

  // Generate connections
  const connections = useMemo(() => {
    const conns = [];

    // Input to hidden (W^T)
    layers.inputLayer.forEach(input => {
      layers.hiddenLayer.forEach(hidden => {
        const isHighlighted = highlightedFeature !== null &&
                            input.index === highlightedFeature;
        conns.push({
          start: input,
          end: hidden,
          type: 'W_transpose',
          highlighted: isHighlighted,
          weight: W ? W[input.index][hidden.index] : 0.5
        });
      });
    });

    // Hidden to output (W)
    layers.hiddenLayer.forEach(hidden => {
      layers.outputLayer.forEach(output => {
        const isHighlighted = highlightedFeature !== null &&
                            output.index === highlightedFeature;
        conns.push({
          start: hidden,
          end: output,
          type: 'W',
          highlighted: isHighlighted,
          weight: W ? W[output.index][hidden.index] : 0.5
        });
      });
    });

    return conns;
  }, [layers, W, highlightedFeature]);

  // Calculate bounding boxes for each layer
  const layerBoxes = useMemo(() => {
    const inputSpacing = 0.8;
    const hiddenSpacing = 1.5;
    const outputSpacing = 0.8;
    const layerSpacing = 3;
    const padding = 0.5;

    return {
      input: {
        x: -layerSpacing,
        y: 0,
        width: 0.8,
        height: (inputDim - 1) * inputSpacing + padding * 2
      },
      hidden: {
        x: 0,
        y: 0,
        width: 0.8,
        height: (hiddenDim - 1) * hiddenSpacing + padding * 2
      },
      output: {
        x: layerSpacing,
        y: 0,
        width: 0.8,
        height: (inputDim - 1) * outputSpacing + padding * 2
      }
    };
  }, [inputDim, hiddenDim]);

  return (
    <group>
      {/* Layer Boxes */}
      {/* Input Layer Box */}
      <mesh position={[layerBoxes.input.x, layerBoxes.input.y, -0.1]}>
        <planeGeometry args={[layerBoxes.input.width, layerBoxes.input.height]} />
        <meshBasicMaterial
          color="#FFFFFF"
          opacity={0.2}
          transparent
          side={THREE.DoubleSide}
        />
      </mesh>
      <lineSegments position={[layerBoxes.input.x, layerBoxes.input.y, -0.05]}>
        <edgesGeometry args={[new THREE.PlaneGeometry(layerBoxes.input.width, layerBoxes.input.height)]} />
        <lineBasicMaterial color="#FFFFFF" opacity={0.2} transparent />
      </lineSegments>

      {/* Hidden Layer Box */}
      <mesh position={[layerBoxes.hidden.x, layerBoxes.hidden.y, -0.1]}>
        <planeGeometry args={[layerBoxes.hidden.width, layerBoxes.hidden.height]} />
        <meshBasicMaterial
          color="#FFFFFF"
          opacity={0.2}
          transparent
          side={THREE.DoubleSide}
        />
      </mesh>
      <lineSegments position={[layerBoxes.hidden.x, layerBoxes.hidden.y, -0.05]}>
        <edgesGeometry args={[new THREE.PlaneGeometry(layerBoxes.hidden.width, layerBoxes.hidden.height)]} />
        <lineBasicMaterial color="#FFFFFF" opacity={0.2} transparent />
      </lineSegments>

      {/* Output Layer Box */}
      <mesh position={[layerBoxes.output.x, layerBoxes.output.y, -0.1]}>
        <planeGeometry args={[layerBoxes.output.width, layerBoxes.output.height]} />
        <meshBasicMaterial
          color="#FFFFFF"
          opacity={0.2}
          transparent
          side={THREE.DoubleSide}
        />
      </mesh>
      <lineSegments position={[layerBoxes.output.x, layerBoxes.output.y, -0.05]}>
        <edgesGeometry args={[new THREE.PlaneGeometry(layerBoxes.output.width, layerBoxes.output.height)]} />
        <lineBasicMaterial color="#FFFFFF" opacity={0.2} transparent />
      </lineSegments>

      {/* Input Layer Neurons */}
      {layers.inputLayer.map((neuron, i) => (
        <mesh key={`input-${i}`} position={[neuron.x, neuron.y, neuron.z]}>
          <sphereGeometry args={[0.15, 32, 32]} />
          <meshStandardMaterial
            color={highlightedFeature === i ? '#B8860B' : '#FFFFFF'}
            emissive={highlightedFeature === i ? '#B8860B' : '#000000'}
            emissiveIntensity={highlightedFeature === i ? 0.3 : 0}
          />
        </mesh>
      ))}

      {/* Hidden Layer Neurons */}
      {layers.hiddenLayer.map((neuron, i) => (
        <mesh key={`hidden-${i}`} position={[neuron.x, neuron.y, neuron.z]}>
          <sphereGeometry args={[0.2, 32, 32]} />
          <meshStandardMaterial color="#FFFFFF" />
        </mesh>
      ))}

      {/* Output Layer Neurons */}
      {layers.outputLayer.map((neuron, i) => (
        <mesh key={`output-${i}`} position={[neuron.x, neuron.y, neuron.z]}>
          <sphereGeometry args={[0.15, 32, 32]} />
          <meshStandardMaterial
            color={highlightedFeature === i ? '#B8860B' : '#FFFFFF'}
            emissive={highlightedFeature === i ? '#B8860B' : '#000000'}
            emissiveIntensity={highlightedFeature === i ? 0.3 : 0}
          />
        </mesh>
      ))}

      {/* Connections */}
      {connections.map((conn, i) => {
        const curve = new THREE.QuadraticBezierCurve3(
          new THREE.Vector3(conn.start.x, conn.start.y, conn.start.z),
          new THREE.Vector3(
            (conn.start.x + conn.end.x) / 2,
            (conn.start.y + conn.end.y) / 2,
            0.3
          ),
          new THREE.Vector3(conn.end.x, conn.end.y, conn.end.z)
        );

        const points = curve.getPoints(50);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        const opacity = conn.highlighted ? 0.8 : 0.25;
        const lineWidth = conn.highlighted ? 3 : 1;
        const color = conn.highlighted ? '#B8860B' : '#FFFFFF';

        return (
          <line key={`conn-${i}`} geometry={geometry}>
            <lineBasicMaterial
              color={color}
              opacity={opacity}
              transparent
              linewidth={lineWidth}
            />
          </line>
        );
      })}

      {/* Layer Labels */}
      <group position={[-3, -3, 0]}>
        <Text
          text="x (Input)"
          position={[0, 0, 0]}
          size={0.3}
          color="#4A90E2"
        />
      </group>
      <group position={[0, -3, 0]}>
        <Text
          text="h (Hidden)"
          position={[0, 0, 0]}
          size={0.3}
          color="#E74C3C"
        />
      </group>
      <group position={[3, -3, 0]}>
        <Text
          text="x' (Output)"
          position={[0, 0, 0]}
          size={0.3}
          color="#2ECC71"
        />
      </group>

      {/* ReLU label */}
      <group position={[3.5, 3, 0]}>
        <Text
          text="ReLU"
          position={[0, 0, 0]}
          size={0.25}
          color="#FF6B6B"
        />
      </group>
    </group>
  );
}

// Simple text component using HTML
function Text({ text, position, size, color }) {
  return null; // We'll add proper text rendering later with drei's Text component
}
