import React, { useState, useMemo, useRef, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { Text, Line, Html } from "@react-three/drei";

/**
 * Polytope geometry definitions
 * Each polytope has vertices, edges, symmetry group, and color
 */
const POLYTOPES = {
  digon: {
    name: "Digon",
    p: 2,
    color: "#FFD700", // yellow
    vertices: [
      [0, 1, 0],
      [0, -1, 0]
    ],
    edges: [[0, 1]],
    symmetryGroup: [
      [0, 1], // identity
      [1, 0]  // 180° rotation
    ],
    groupNames: ["e (identity)", "r (180°)"]
  },
  triangle: {
    name: "Triangle",
    p: 3,
    color: "#00FF00", // green
    vertices: [
      [0, 1, 0],
      [Math.sqrt(3)/2, -0.5, 0],
      [-Math.sqrt(3)/2, -0.5, 0]
    ],
    edges: [[0, 1], [1, 2], [2, 0]],
    symmetryGroup: [
      [0, 1, 2], [1, 2, 0], [2, 0, 1],
      [0, 2, 1], [2, 1, 0], [1, 0, 2]
    ],
    groupNames: [
      "e (identity)", "r (120°)", "r² (240°)",
      "s (reflect-0)", "sr (reflect-1)", "sr² (reflect-2)"
    ]
  },
  tetrahedron: {
    name: "Tetrahedron",
    p: 4,
    color: "#0066FF", // blue
    vertices: [
      [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ].map(v => {
      const norm = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
      return [v[0]/norm, v[1]/norm, v[2]/norm];
    }),
    edges: [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
    symmetryGroup: [
      [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2],
      [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]
    ],
    groupNames: [
      "e (identity)", "r₁ (120° v0)", "r₁² (240° v0)",
      "r₂ (180° e01)", "r₃ (180°)", "σ (composite)"
    ]
  },
  pentagon: {
    name: "Pentagon",
    p: 5,
    color: "#FF8C00", // orange
    vertices: Array.from({ length: 5 }, (_, i) => {
      const angle = (2 * Math.PI * i) / 5 - Math.PI / 2;
      return [Math.cos(angle), Math.sin(angle), 0];
    }),
    edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
    symmetryGroup: [
      [0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1],
      [3, 4, 0, 1, 2], [4, 0, 1, 2, 3], [0, 4, 3, 2, 1],
      [1, 0, 4, 3, 2], [2, 1, 0, 4, 3], [3, 2, 1, 0, 4],
      [4, 3, 2, 1, 0]
    ],
    groupNames: [
      "e (identity)", "r (72°)", "r² (144°)", "r³ (216°)", "r⁴ (288°)",
      "s₀ (reflect-0)", "s₁ (reflect-1)", "s₂ (reflect-2)",
      "s₃ (reflect-3)", "s₄ (reflect-4)"
    ]
  },
  squareAntiprism: {
    name: "Square Antiprism",
    p: 8,
    color: "#9370DB", // purple
    vertices: [
      [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
      [Math.SQRT2/2, 0, -1], [0, -Math.SQRT2/2, -1],
      [-Math.SQRT2/2, 0, -1], [0, Math.SQRT2/2, -1]
    ].map(v => {
      const norm = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
      return [v[0]/norm * 1.2, v[1]/norm * 1.2, v[2]/norm * 1.2];
    }),
    edges: [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [0, 7], [1, 4], [1, 5], [2, 5], [2, 6], [3, 6], [3, 7]
    ],
    symmetryGroup: [
      [0, 1, 2, 3, 4, 5, 6, 7],
      [1, 2, 3, 0, 5, 6, 7, 4],
      [2, 3, 0, 1, 6, 7, 4, 5],
      [3, 0, 1, 2, 7, 4, 5, 6],
      [3, 2, 1, 0, 7, 6, 5, 4],
      [0, 3, 2, 1, 4, 7, 6, 5]
    ],
    groupNames: [
      "e (identity)", "r (90°)", "r² (180°)",
      "r³ (270°)", "σ₁ (reflect-1)", "σ₂ (reflect-2)"
    ]
  },
  isotropic: {
    name: "Isotropic O(n)",
    p: 20,
    color: "#808080", // grey
    vertices: Array.from({ length: 20 }, () => {
      const theta = Math.random() * 2 * Math.PI;
      const phi = Math.acos(2 * Math.random() - 1);
      return [
        Math.sin(phi) * Math.cos(theta),
        Math.sin(phi) * Math.sin(theta),
        Math.cos(phi)
      ];
    }),
    edges: [],
    symmetryGroup: [[...Array(20).keys()]],
    groupNames: ["Continuous O(n) rotations"],
    isIsotropic: true
  }
};

/**
 * Apply a permutation to vertices
 */
function applyPermutation(vertices, permutation) {
  return permutation.map(i => vertices[i]);
}

/**
 * Individual polytope visualization component
 */
function PolytopeView({ polytope, currentTransform, showLabels = true }) {
  const meshRef = useRef();

  const transformedVertices = useMemo(() => {
    if (!currentTransform) return polytope.vertices;
    return applyPermutation(polytope.vertices, currentTransform);
  }, [polytope, currentTransform]);

  useFrame((state) => {
    if (polytope.isIsotropic && meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.3;
      meshRef.current.rotation.x = state.clock.elapsedTime * 0.2;
    }
  });

  return (
    <group ref={meshRef}>
      {/* Edges */}
      {polytope.edges.map((edge, idx) => {
        const start = transformedVertices[edge[0]];
        const end = transformedVertices[edge[1]];
        return (
          <Line
            key={`edge-${idx}`}
            points={[start, end]}
            color={polytope.color}
            lineWidth={2}
          />
        );
      })}

      {/* Vertices */}
      {transformedVertices.map((vertex, idx) => (
        <group key={`vertex-${idx}`}>
          <mesh position={vertex}>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshStandardMaterial color={polytope.color} />
          </mesh>

          {showLabels && !polytope.isIsotropic && (
            <Text
              position={[vertex[0] * 1.3, vertex[1] * 1.3, vertex[2] * 1.3]}
              fontSize={0.15}
              color="white"
              anchorX="center"
              anchorY="middle"
            >
              {idx}
            </Text>
          )}
        </group>
      ))}

      {/* Transformation arrows */}
      {currentTransform && !polytope.isIsotropic && (
        <>
          {currentTransform.map((targetIdx, sourceIdx) => {
            if (sourceIdx === targetIdx) return null;

            const start = polytope.vertices[sourceIdx];
            const end = transformedVertices[sourceIdx];
            const mid = [
              (start[0] + end[0]) / 2,
              (start[1] + end[1]) / 2,
              (start[2] + end[2]) / 2
            ];

            return (
              <group key={`arrow-${sourceIdx}`}>
                <Line
                  points={[start, end]}
                  color="#FF00FF"
                  lineWidth={1}
                  dashed
                  dashScale={50}
                  dashSize={0.02}
                  gapSize={0.01}
                />
                <mesh position={mid}>
                  <sphereGeometry args={[0.04, 8, 8]} />
                  <meshStandardMaterial color="#FF00FF" />
                </mesh>
              </group>
            );
          })}
        </>
      )}
    </group>
  );
}

/**
 * Control Panel Component
 */
function ControlPanel({
  selectedPolytope,
  setSelectedPolytope,
  currentTransformIdx,
  setCurrentTransformIdx,
  isAnimating,
  setIsAnimating,
  animationSpeed,
  setAnimationSpeed,
  polytope,
  currentTransform,
  handlePrev,
  handleNext
}) {
  return (
    <Html fullscreen>
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        pointerEvents: 'auto',
        zIndex: 1000
      }}>
        <div style={{
          background: 'rgba(0, 0, 0, 0.9)',
          padding: '20px',
          borderRadius: '10px',
          color: 'white',
          fontFamily: 'monospace',
          fontSize: '14px',
          border: '1px solid #444',
          width: '320px'
        }}>
          <h3 style={{
            margin: '0 0 15px 0',
            fontSize: '16px',
            borderBottom: '1px solid #555',
            paddingBottom: '10px'
          }}>
            Group Action Visualization
          </h3>

          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px', color: '#aaa' }}>
              Polytope:
            </label>
            <select
              value={selectedPolytope}
              onChange={(e) => {
                setSelectedPolytope(e.target.value);
                setCurrentTransformIdx(0);
                setIsAnimating(false);
              }}
              style={{
                width: '100%',
                padding: '8px',
                borderRadius: '5px',
                background: '#333',
                color: 'white',
                border: '1px solid #555',
                cursor: 'pointer'
              }}
            >
              {Object.entries(POLYTOPES).map(([key, p]) => (
                <option key={key} value={key}>
                  {p.name} (p={p.p})
                </option>
              ))}
            </select>
          </div>

          <div style={{
            marginBottom: '15px',
            padding: '10px',
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '5px'
          }}>
            <div style={{ fontSize: '11px', color: '#aaa', marginBottom: '3px' }}>
              Transformation {currentTransformIdx + 1}/{polytope.symmetryGroup.length}:
            </div>
            <div style={{ fontSize: '13px', fontWeight: 'bold', color: polytope.color }}>
              {polytope.groupNames[currentTransformIdx]}
            </div>
            <div style={{ fontSize: '10px', color: '#888', marginTop: '5px' }}>
              π = [{currentTransform.join(', ')}]
            </div>
          </div>

          <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
            <button onClick={handlePrev} style={{
              flex: 1, padding: '8px', borderRadius: '5px',
              background: '#444', color: 'white', border: 'none', cursor: 'pointer'
            }}>← Prev</button>
            <button onClick={() => setIsAnimating(!isAnimating)} style={{
              flex: 1, padding: '8px', borderRadius: '5px',
              background: isAnimating ? '#c44' : '#4c4',
              color: 'white', border: 'none', cursor: 'pointer'
            }}>{isAnimating ? '⏸ Pause' : '▶ Play'}</button>
            <button onClick={handleNext} style={{
              flex: 1, padding: '8px', borderRadius: '5px',
              background: '#444', color: 'white', border: 'none', cursor: 'pointer'
            }}>Next →</button>
          </div>

          <div style={{ marginBottom: '10px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontSize: '11px', color: '#aaa' }}>
              Animation Speed: {animationSpeed.toFixed(1)}s
            </label>
            <input
              type="range"
              min="0.5"
              max="3"
              step="0.1"
              value={animationSpeed}
              onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          <div style={{
            fontSize: '10px',
            color: '#888',
            marginTop: '15px',
            paddingTop: '15px',
            borderTop: '1px solid #444'
          }}>
            <strong>Symmetry Group:</strong> {polytope.isIsotropic ? 'O(n)' : `|Γ| = ${polytope.symmetryGroup.length}`}
            <br />
            {polytope.isIsotropic ? (
              <span style={{ color: '#aaa' }}>Continuous rotational symmetry</span>
            ) : (
              <>
                <strong>Vertices:</strong> V = {`{0, ..., ${polytope.p - 1}}`}
                <br />
                <span style={{ color: '#aaa' }}>Magenta dashed lines show vertex mappings</span>
              </>
            )}
          </div>
        </div>
      </div>
    </Html>
  );
}

/**
 * Main component
 */
export default function GroupActionVisualization() {
  const [selectedPolytope, setSelectedPolytope] = useState("triangle");
  const [currentTransformIdx, setCurrentTransformIdx] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1.5);

  const polytope = POLYTOPES[selectedPolytope];
  const currentTransform = useMemo(() => {
    return polytope.symmetryGroup[currentTransformIdx];
  }, [polytope, currentTransformIdx]);

  useEffect(() => {
    if (!isAnimating) return;

    const interval = setInterval(() => {
      setCurrentTransformIdx((prev) =>
        (prev + 1) % polytope.symmetryGroup.length
      );
    }, animationSpeed * 1000);

    return () => clearInterval(interval);
  }, [isAnimating, animationSpeed, polytope]);

  const handleNext = () => {
    setCurrentTransformIdx((prev) =>
      (prev + 1) % polytope.symmetryGroup.length
    );
  };

  const handlePrev = () => {
    setCurrentTransformIdx((prev) =>
      (prev - 1 + polytope.symmetryGroup.length) % polytope.symmetryGroup.length
    );
  };

  return (
    <>
      <PolytopeView
        polytope={polytope}
        currentTransform={currentTransform}
        showLabels={!polytope.isIsotropic}
      />
      <ControlPanel
        selectedPolytope={selectedPolytope}
        setSelectedPolytope={setSelectedPolytope}
        currentTransformIdx={currentTransformIdx}
        setCurrentTransformIdx={setCurrentTransformIdx}
        isAnimating={isAnimating}
        setIsAnimating={setIsAnimating}
        animationSpeed={animationSpeed}
        setAnimationSpeed={setAnimationSpeed}
        polytope={polytope}
        currentTransform={currentTransform}
        handlePrev={handlePrev}
        handleNext={handleNext}
      />
    </>
  );
}
