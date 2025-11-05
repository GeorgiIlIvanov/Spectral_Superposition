/**
 * Colorbar mapping utilities for feature geometry visualization
 * Maps RGB colors to fractional dimensionality values
 *
 * Color mapping from the feature geometry image:
 * - Dark Red (0.0-0.3): No feature / very low dimensionality
 * - Orange (0.4): Pentagon (2/5)
 * - Yellow (0.5): Digon (1/2)
 * - Green (0.667): Triangle (2/3)
 * - Blue (0.75): Tetrahedron (3/4)
 */

// High-fidelity color lookup table
const COLOR_LUT = [
  // Very low values (0.0 - 0.35): Dark Red -> Red (no feature region)
  { value: 0.0, rgb: [120, 40, 50] },      // Very dark red
  { value: 0.05, rgb: [140, 50, 55] },     // Dark red
  { value: 0.1, rgb: [160, 60, 60] },      // Dark red
  { value: 0.15, rgb: [180, 70, 65] },     // Red
  { value: 0.2, rgb: [200, 80, 70] },      // Red
  { value: 0.25, rgb: [220, 90, 75] },     // Light red
  { value: 0.3, rgb: [240, 100, 80] },     // Orange-red
  { value: 0.35, rgb: [250, 120, 85] },    // Red-orange

  // Square Antiprism region (3/8 = 0.375): Purple/Pink
  { value: 0.375, rgb: [199, 125, 223] },  // Purple (Square Antiprism - 3/8)

  // Pentagon region (2/5 = 0.4): Orange
  { value: 0.4, rgb: [255, 160, 122] },    // Orange (Pentagon - 2/5)
  { value: 0.42, rgb: [255, 170, 110] },   // Orange
  { value: 0.44, rgb: [255, 180, 100] },   // Orange-yellow
  { value: 0.46, rgb: [255, 190, 90] },    // Yellow-orange
  { value: 0.48, rgb: [255, 200, 80] },    // Light orange

  // Digon/Octahedron region (1/2 = 0.5): Yellow
  { value: 0.5, rgb: [255, 215, 0] },      // Gold/Yellow (Digon/Octahedron - 1/2)
  { value: 0.52, rgb: [250, 220, 70] },    // Yellow
  { value: 0.54, rgb: [240, 225, 90] },    // Light yellow
  { value: 0.56, rgb: [230, 230, 110] },   // Yellow-green
  { value: 0.58, rgb: [210, 230, 120] },   // Yellow-green
  { value: 0.60, rgb: [180, 220, 130] },   // Light green
  { value: 0.62, rgb: [150, 210, 140] },   // Green
  { value: 0.64, rgb: [120, 200, 145] },   // Green

  // Triangle region (2/3 ≈ 0.667): Green
  { value: 0.667, rgb: [82, 183, 136] },   // Green (Triangle - 2/3)
  { value: 0.68, rgb: [75, 185, 140] },    // Green
  { value: 0.70, rgb: [70, 190, 160] },    // Green-cyan
  { value: 0.72, rgb: [65, 195, 180] },    // Cyan-green

  // Tetrahedron region (3/4 = 0.75): Blue
  { value: 0.74, rgb: [70, 200, 195] },    // Cyan
  { value: 0.75, rgb: [78, 205, 196] },    // Cyan-blue (Tetrahedron - 3/4)
  { value: 0.76, rgb: [75, 200, 200] },    // Cyan-blue
  { value: 0.78, rgb: [70, 190, 210] },    // Blue-cyan
  { value: 0.80, rgb: [65, 180, 220] },    // Light blue
  { value: 0.82, rgb: [60, 170, 230] },    // Blue
  { value: 0.85, rgb: [55, 160, 240] },    // Blue
  { value: 0.90, rgb: [50, 140, 250] },    // Deep blue
  { value: 0.95, rgb: [45, 120, 255] },    // Navy blue
  { value: 1.0, rgb: [40, 100, 255] },     // Dark blue
];

/**
 * Convert RGB color to fractional dimensionality value
 * Uses nearest neighbor search in RGB space
 * @param {number} r - Red component (0-255)
 * @param {number} g - Green component (0-255)
 * @param {number} b - Blue component (0-255)
 * @returns {number} - Fractional dimensionality value (0-1)
 */
export function rgbToValue(r, g, b) {
  let minDistance = Infinity;
  let closestValue = 0.5;

  for (const entry of COLOR_LUT) {
    const [lr, lg, lb] = entry.rgb;
    const distance = Math.sqrt(
      Math.pow(r - lr, 2) +
      Math.pow(g - lg, 2) +
      Math.pow(b - lb, 2)
    );

    if (distance < minDistance) {
      minDistance = distance;
      closestValue = entry.value;
    }
  }

  return closestValue;
}

/**
 * Convert fractional dimensionality value to RGB color
 * @param {number} value - Fractional dimensionality (0-1)
 * @returns {Array<number>} - RGB values [r, g, b]
 */
export function valueToRGB(value) {
  value = Math.max(0, Math.min(1, value)); // Clamp to [0, 1]

  // Find the two nearest LUT entries
  let i0 = 0;
  for (let i = 0; i < COLOR_LUT.length - 1; i++) {
    if (value >= COLOR_LUT[i].value && value <= COLOR_LUT[i + 1].value) {
      i0 = i;
      break;
    }
  }

  const i1 = Math.min(i0 + 1, COLOR_LUT.length - 1);
  const v0 = COLOR_LUT[i0].value;
  const v1 = COLOR_LUT[i1].value;
  const rgb0 = COLOR_LUT[i0].rgb;
  const rgb1 = COLOR_LUT[i1].rgb;

  // Linear interpolation
  const t = v1 === v0 ? 0 : (value - v0) / (v1 - v0);

  return [
    Math.round(rgb0[0] * (1 - t) + rgb1[0] * t),
    Math.round(rgb0[1] * (1 - t) + rgb1[1] * t),
    Math.round(rgb0[2] * (1 - t) + rgb1[2] * t)
  ];
}

/**
 * Map fractional dimensionality to the nearest available geometry
 * @param {number} fractionalDim - Fractional dimensionality (0-1)
 * @returns {Object} - {numFeatures, name, exactFractionalDim}
 */
export function fractionalDimToGeometry(fractionalDim) {
  const geometries = [
    { fractionalDim: 0.375, numFeatures: 8, name: 'Square Antiprism' },      // 3/8
    { fractionalDim: 0.4, numFeatures: 5, name: 'Triangular Dipyramid' },    // 2/5
    { fractionalDim: 0.5, numFeatures: 2, name: 'Digon' },                   // 1/2
    { fractionalDim: 0.5, numFeatures: 6, name: 'Octahedron' },              // 1/2
    { fractionalDim: 2/3, numFeatures: 3, name: 'Triangle' },                // 2/3
    { fractionalDim: 0.75, numFeatures: 4, name: 'Tetrahedron' },            // 3/4
  ];

  let closest = geometries[0];
  let minDiff = Math.abs(fractionalDim - closest.fractionalDim);

  for (const geom of geometries) {
    const diff = Math.abs(fractionalDim - geom.fractionalDim);
    if (diff < minDiff) {
      minDiff = diff;
      closest = geom;
    }
  }

  return {
    numFeatures: closest.numFeatures,
    name: closest.name,
    exactFractionalDim: closest.fractionalDim
  };
}
