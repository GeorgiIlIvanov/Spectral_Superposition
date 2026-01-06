# Group Action Visualizations for Superposition Theory

## Overview

This document describes the interactive 3D visualizations of symmetry group actions on polytopes, created to illustrate the theoretical framework in our paper on superposition.

## Mathematical Framework

### Symmetry Groups and Polytopes

For a feature cluster $C$ of size $|C| = p$, we define:

- **Vertex Set**: $V = \{0, 1, \ldots, p-1\}$
- **Symmetry Group**: $\Gamma \subseteq S_p$ acting transitively on $V$
- **Group Action**: $\gamma \cdot (i, j) = (\gamma(i), \gamma(j))$ for $\gamma \in \Gamma$
- **Permutation Matrix**: $(P_\gamma)_{ij} = \delta_{i, \gamma(j)}$

### Orbit Decomposition

The group action partitions $V \times V$ into $R$ disjoint orbits:
$$V \times V = \mathcal{O}_1 \sqcup \mathcal{O}_2 \sqcup \cdots \sqcup \mathcal{O}_R$$

where $R = |G : G_s|$ by the orbit-stabilizer theorem.

## Implemented Polytopes

### 1. Digon (p=2) - Yellow

**Geometry**: Two vertices connected by edges
- **Vertices**: $V = \{0, 1\}$
- **Symmetry Group**: $D_2 = \{e, r\}$ (dihedral group of order 2)
  - $e$: identity
  - $r$: 180° rotation (vertex swap)
- **Group Order**: $|\Gamma| = 2$

**Mathematical Significance**: Simplest case showing binary feature interactions

### 2. Triangle (p=3) - Green

**Geometry**: Equilateral triangle
- **Vertices**: $V = \{0, 1, 2\}$ positioned at 120° intervals
- **Symmetry Group**: $D_3$ (dihedral group of order 6)
  - 3 rotations: $\{e, r, r^2\}$ (0°, 120°, 240°)
  - 3 reflections: $\{s, sr, sr^2\}$
- **Group Order**: $|\Gamma| = 6$

**Mathematical Significance**: First non-trivial case with both rotational and reflective symmetry

### 3. Tetrahedron (p=4) - Blue

**Geometry**: Regular tetrahedron inscribed in sphere
- **Vertices**: $V = \{0, 1, 2, 3\}$ at tetrahedral vertices
- **Symmetry Group**: Tetrahedral group $T$ (subset shown)
  - Rotations around vertices (120°, 240°)
  - Rotations around edges (180°)
- **Group Order**: Full group $|T| = 12$ (subset of 6 shown for clarity)

**Mathematical Significance**: First fully 3D polytope, demonstrates face-centered symmetries

### 4. Pentagon (p=5) - Orange

**Geometry**: Regular pentagon
- **Vertices**: $V = \{0, 1, 2, 3, 4\}$ at 72° intervals
- **Symmetry Group**: $D_5$ (dihedral group of order 10)
  - 5 rotations: multiples of 72°
  - 5 reflections through vertices
- **Group Order**: $|\Gamma| = 10$

**Mathematical Significance**: Prime order case, demonstrates aperiodic structure

### 5. Square Antiprism (p=8) - Purple

**Geometry**: Two parallel squares with vertices offset by 45°
- **Vertices**: $V = \{0, \ldots, 7\}$
  - Top square: vertices 0-3
  - Bottom square: vertices 4-7 (rotated 45°)
- **Symmetry Group**: $D_{4d}$ (antiprism group, subset shown)
  - Rotations: 90°, 180°, 270°
  - Various reflections
- **Group Order**: Full group $|D_{4d}| = 16$ (subset of 6 shown)

**Mathematical Significance**: Demonstrates chiral symmetry breaking in 3D

### 6. Isotropic O(n) (p=20) - Grey

**Geometry**: Random points on unit sphere
- **Vertices**: 20 randomly distributed points on $S^2$
- **Symmetry Group**: Continuous $O(n)$ (orthogonal group)
  - Continuous rotations (not discrete)
  - No reflective structure
- **Mathematical Significance**: Limiting case as $p \to \infty$, represents fully isotropic features

## Visualization Features

### Interactive Controls

1. **Polytope Selector**: Switch between different geometric structures
2. **Transform Navigation**:
   - **Next/Prev**: Step through group elements
   - **Play/Pause**: Automatic animation through all transformations
   - **Speed Control**: Adjust animation speed (0.5s - 3.0s per transform)

### Visual Elements

1. **Vertices**:
   - Colored spheres at polytope vertices
   - Labeled 0 to p-1 (except isotropic case)
   - Size: 0.08 units

2. **Edges**:
   - Solid lines connecting vertices
   - Colored according to polytope type

3. **Transformation Arrows** (magenta dashed lines):
   - Show vertex mappings under current group element
   - Only visible for non-identity transformations
   - Midpoint marker indicates direction

4. **Information Display**:
   - Current transformation name (e.g., "r² (240°)")
   - Permutation representation: $\pi = [...]$
   - Group order: $|\Gamma|$
   - Transformation index

### Color Coding

| Polytope | Color | Hex Code | Purpose |
|----------|-------|----------|---------|
| Digon | Yellow | `#FFD700` | Binary interactions |
| Triangle | Green | `#00FF00` | Ternary symmetry |
| Tetrahedron | Blue | `#0066FF` | 3D face symmetry |
| Pentagon | Orange | `#FF8C00` | Prime-order groups |
| Square Antiprism | Purple | `#9370DB` | Chiral structures |
| Isotropic O(n) | Grey | `#808080` | Continuous limit |

## Implementation Details

### Technology Stack

- **React Three Fiber**: 3D rendering in React
- **Three.js**: WebGL graphics library
- **@react-three/drei**: Utility components (Text, Line, OrbitControls)

### Key Components

1. **PolytopeView**: Renders individual polytope with transformations
2. **GroupActionVisualization**: Main component with controls
3. **POLYTOPES**: Configuration object defining all geometries

### Mathematical Implementation

#### Vertex Positioning

Vertices are positioned on unit sphere using:
- **2D polytopes**: Polar coordinates $(r, \theta)$ in xy-plane
- **3D polytopes**: Normalized Cartesian coordinates

#### Permutation Application

```javascript
function applyPermutation(vertices, permutation) {
  return permutation.map(i => vertices[i]);
}
```

For permutation $\pi \in S_p$, maps vertex $i$ to position of vertex $\pi(i)$.

#### Continuous Rotation (Isotropic Case)

Uses time-based rotation matrices:
```javascript
rotation.y = time * 0.3
rotation.x = time * 0.2
```

## Usage in Paper

### Suggested Figure Captions

**Figure X**: *Interactive 3D visualizations of symmetry group actions on polytopes. Each polytope represents a feature cluster of size p with symmetry group Γ ⊆ Sₚ. Vertices are labeled 0 to p-1, and transformations show how group elements permute the vertex set.*

**Figure Y**: *Visualization of group action γ·(i,j) = (γ(i), γ(j)) on vertex pairs. Magenta dashed lines indicate vertex mappings under the current group element, with the permutation matrix Pᵧ shown in the interface.*

### Key Insights for Paper

1. **Transitive Action**: All polytopes demonstrate $\Gamma$ acting transitively on $V$
2. **Orbit Structure**: Different polytope types have different orbit decompositions
3. **Dimension Scaling**: Complexity grows from 2D (digon, triangle, pentagon) to 3D (tetrahedron, antiprism)
4. **Isotropic Limit**: O(n) case shows continuous symmetry as $p \to \infty$

## Accessing the Visualization

1. **Development Mode**:
   ```bash
   npm install
   npm run dev
   ```
   Navigate to `http://localhost:5173/`

2. **In Application**:
   - Click on "Group Actions on Polytopes (Γ ⊆ Sₚ)" section
   - Use dropdown to select polytope
   - Click Play or use Next/Prev to cycle through transformations

## Mathematical Extensions

### Future Enhancements

1. **Orbit Visualization**: Color edges by orbit $\mathcal{O}_i$
2. **Cayley Graphs**: Show group structure as graph
3. **Representation Theory**: Visualize irreducible representations
4. **Permutation Matrices**: Display $(P_\gamma)_{ij}$ as heatmap

## References

- **Cayley's Theorem**: Every group $\Gamma$ is isomorphic to a subgroup of $S_p$
- **Orbit-Stabilizer Theorem**: $|\mathcal{O}_i| \cdot |G_s| = |G|$
- **Burnside's Lemma**: Counts orbits using fixed points

## File Locations

- **Component**: `src/components/GroupActionVisualization.jsx`
- **Integration**: `src/App.jsx` (lines 7, 27, 51, 186-209)
- **Documentation**: `GROUP_ACTION_VISUALIZATIONS.md` (this file)

---

**Authors**: Implementation for spectral superposition paper
**Date**: December 2024
**Framework**: React + Three.js + React Three Fiber
