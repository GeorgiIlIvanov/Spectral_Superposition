/**
 * Utility functions for Toy Models of Superposition
 */

/**
 * Generate feature matrix W (n x m) for superposition model
 * @param {number} n - Input/output dimension
 * @param {number} m - Hidden dimension (bottleneck)
 * @returns {Array<Array<number>>} - Feature matrix W
 */
export function generateFeatureMatrix(n, m) {
  const W = [];
  for (let i = 0; i < n; i++) {
    W[i] = [];
    for (let j = 0; j < m; j++) {
      W[i][j] = Math.random() * 2 - 1; // Random initialization [-1, 1]
    }
  }
  return normalizeColumns(W);
}

/**
 * Normalize columns of matrix to unit vectors
 * @param {Array<Array<number>>} W
 * @returns {Array<Array<number>>}
 */
function normalizeColumns(W) {
  const n = W.length;
  const m = W[0].length;

  for (let j = 0; j < m; j++) {
    let norm = 0;
    for (let i = 0; i < n; i++) {
      norm += W[i][j] * W[i][j];
    }
    norm = Math.sqrt(norm);

    for (let i = 0; i < n; i++) {
      W[i][j] /= norm;
    }
  }
  return W;
}

/**
 * Compute W^T W (Gram matrix)
 * @param {Array<Array<number>>} W - Feature matrix (n x m)
 * @returns {Array<Array<number>>} - Gram matrix (n x n)
 */
export function computeGramMatrix(W) {
  const n = W.length;
  const m = W[0].length;
  const WTW = [];

  // W^T is m x n, W is n x m, so W^T W is m x m
  // But for visualization we want n x n (feature-to-feature similarity)
  // So we compute W W^T instead

  for (let i = 0; i < n; i++) {
    WTW[i] = [];
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < m; k++) {
        sum += W[i][k] * W[j][k];
      }
      WTW[i][j] = sum;
    }
  }

  return WTW;
}

/**
 * ReLU activation function
 * @param {number} x
 * @returns {number}
 */
function relu(x) {
  return Math.max(0, x);
}

/**
 * Forward pass: f(W, x) = ReLU(W^T W x + b)
 * @param {Array<Array<number>>} W - Feature matrix (n x m)
 * @param {Array<number>} x - Input vector (n-dimensional)
 * @param {Array<number>} b - Bias vector (n-dimensional)
 * @returns {Object} - {h: hidden activation, xPrime: output}
 */
export function forwardPass(W, x, b = null) {
  const n = W.length;
  const m = W[0].length;

  if (!b) {
    b = new Array(n).fill(0);
  }

  // h = W^T x (m-dimensional)
  const h = [];
  for (let j = 0; j < m; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += W[i][j] * x[i];
    }
    h[j] = sum;
  }

  // x' = W h + b (n-dimensional)
  const xPrime = [];
  for (let i = 0; i < n; i++) {
    let sum = b[i];
    for (let j = 0; j < m; j++) {
      sum += W[i][j] * h[j];
    }
    xPrime[i] = relu(sum);
  }

  return { h, xPrime };
}

/**
 * Generate positions for Thomson problem solutions on a sphere
 * Based on common configurations from the paper
 * @param {number} numFeatures - Number of features
 * @returns {Array<{x, y, z}>} - Positions on unit sphere
 */
export function getThomsonPositions(numFeatures) {
  const positions = [];

  switch(numFeatures) {
    case 2: // Digon (antipodal points)
      positions.push({ x: 0, y: 0, z: 1 });
      positions.push({ x: 0, y: 0, z: -1 });
      break;

    case 3: // Triangle (equilateral on equator)
      for (let i = 0; i < 3; i++) {
        const angle = (i * 2 * Math.PI) / 3;
        positions.push({
          x: Math.cos(angle),
          y: Math.sin(angle),
          z: 0
        });
      }
      break;

    case 4: // Tetrahedron
      positions.push({ x: 1, y: 1, z: 1 });
      positions.push({ x: 1, y: -1, z: -1 });
      positions.push({ x: -1, y: 1, z: -1 });
      positions.push({ x: -1, y: -1, z: 1 });
      break;

    case 5: // Triangular dipyramid
      positions.push({ x: 0, y: 0, z: 1 });
      positions.push({ x: 0, y: 0, z: -1 });
      for (let i = 0; i < 3; i++) {
        const angle = (i * 2 * Math.PI) / 3;
        positions.push({
          x: Math.cos(angle) * 0.866,
          y: Math.sin(angle) * 0.866,
          z: 0
        });
      }
      break;

    case 6: // Octahedron
      positions.push({ x: 1, y: 0, z: 0 });
      positions.push({ x: -1, y: 0, z: 0 });
      positions.push({ x: 0, y: 1, z: 0 });
      positions.push({ x: 0, y: -1, z: 0 });
      positions.push({ x: 0, y: 0, z: 1 });
      positions.push({ x: 0, y: 0, z: -1 });
      break;

    case 8: // Square antiprism
      // Top square
      for (let i = 0; i < 4; i++) {
        const angle = (i * 2 * Math.PI) / 4;
        positions.push({
          x: Math.cos(angle) * 0.7,
          y: Math.sin(angle) * 0.7,
          z: 0.5
        });
      }
      // Bottom square (rotated)
      for (let i = 0; i < 4; i++) {
        const angle = (i * 2 * Math.PI) / 4 + Math.PI / 4;
        positions.push({
          x: Math.cos(angle) * 0.7,
          y: Math.sin(angle) * 0.7,
          z: -0.5
        });
      }
      break;

    default: // Generic spherical distribution
      for (let i = 0; i < numFeatures; i++) {
        const phi = Math.acos(1 - 2 * (i + 0.5) / numFeatures);
        const theta = Math.PI * (1 + Math.sqrt(5)) * i;

        positions.push({
          x: Math.sin(phi) * Math.cos(theta),
          y: Math.sin(phi) * Math.sin(theta),
          z: Math.cos(phi)
        });
      }
  }

  // Normalize all positions to unit sphere
  return positions.map(p => {
    const norm = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    return {
      x: p.x / norm,
      y: p.y / norm,
      z: p.z / norm
    };
  });
}

/**
 * Calculate fractional dimensionality for a feature
 * Based on the formula from the paper
 * @param {number} numFeatures - Number of features (n)
 * @param {number} hiddenDim - Hidden dimension (m)
 * @returns {number} - Fractional dimensionality
 */
export function calculateFractionalDimensionality(numFeatures, hiddenDim) {
  // This is a simplified version - actual computation depends on feature importance
  return hiddenDim / numFeatures;
}

/**
 * Get configuration name and fractional dimensionality for common geometries
 * @returns {Array<{name, numFeatures, fractionalDim, color}>}
 */
export function getGeometricConfigurations() {
  return [
    { name: 'Digon', numFeatures: 2, fractionalDim: 0.5, color: '#FFD700' },          // 1/2
    { name: 'Triangle', numFeatures: 3, fractionalDim: 2/3, color: '#52B788' },       // 2/3 ≈ 0.667
    { name: 'Tetrahedron', numFeatures: 4, fractionalDim: 0.75, color: '#4ECDC4' },   // 3/4
    { name: 'Triangular Dipyramid', numFeatures: 5, fractionalDim: 0.4, color: '#FFA07A' },  // 2/5
    { name: 'Octahedron', numFeatures: 6, fractionalDim: 0.5, color: '#A8E6CF' },     // ~0.5
    { name: 'Square Antiprism', numFeatures: 8, fractionalDim: 0.375, color: '#C77DFF' },  // 3/8
  ];
}

/**
 * Get edge connections for polytope geometries
 * Returns pairs of vertex indices that should be connected
 * @param {number} numFeatures - Number of features
 * @returns {Array<[number, number]>} - Array of edge pairs
 */
export function getPolytopeEdges(numFeatures) {
  switch(numFeatures) {
    case 2: // Digon - single edge
      return [[0, 1]];

    case 3: // Triangle
      return [[0, 1], [1, 2], [2, 0]];

    case 4: // Tetrahedron - all vertices connected
      return [
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3], [2, 3]
      ];

    case 5: // Triangular Dipyramid
      // Top vertex (0) to equator (2,3,4)
      // Bottom vertex (1) to equator (2,3,4)
      // Equator triangle (2,3,4)
      return [
        [0, 2], [0, 3], [0, 4],  // Top to equator
        [1, 2], [1, 3], [1, 4],  // Bottom to equator
        [2, 3], [3, 4], [4, 2]   // Equator triangle
      ];

    case 6: // Octahedron
      return [
        // Equatorial square
        [0, 2], [2, 1], [1, 3], [3, 0],
        // Top connections
        [4, 0], [4, 1], [4, 2], [4, 3],
        // Bottom connections
        [5, 0], [5, 1], [5, 2], [5, 3]
      ];

    case 8: // Square Antiprism
      // Top square: 0,1,2,3
      // Bottom square: 4,5,6,7
      return [
        // Top square
        [0, 1], [1, 2], [2, 3], [3, 0],
        // Bottom square
        [4, 5], [5, 6], [6, 7], [7, 4],
        // Connecting edges (antiprism pattern)
        [0, 4], [0, 5],
        [1, 5], [1, 6],
        [2, 6], [2, 7],
        [3, 7], [3, 4]
      ];

    default:
      return [];
  }
}
