# Bloch Sphere Math Contract

## Input: Density Matrix

The input is a 2x2 density matrix ρ (complex dtype, numpy array).

**Constraints:**
- Shape: (2, 2)
- Hermitian: ρ = ρ†
- Trace: Tr(ρ) = 1
- Positive semidefinite: all eigenvalues ≥ 0

## Computation: Bloch Vector

For a 1-qubit density matrix ρ, the Bloch vector (x, y, z) is computed as:

```
x = Tr(ρ · σ_x)
y = Tr(ρ · σ_y)  
z = Tr(ρ · σ_z)
```

where σ_x, σ_y, σ_z are the Pauli matrices:

```
σ_x = [[0, 1], [1, 0]]
σ_y = [[0, -i], [i, 0]]
σ_z = [[1, 0], [0, -1]]
```

## Output: Bloch Coordinates

Returns a tuple (x, y, z) of real floats.

**Properties:**
- All coordinates are real (imaginary parts should be zero for valid density matrices)
- The vector lies on or inside the Bloch sphere: x² + y² + z² ≤ 1
- Pure states: x² + y² + z² = 1 (on the surface)
- Mixed states: x² + y² + z² < 1 (inside the sphere)

## Special Cases

- |0⟩ state: (0, 0, 1) - north pole
- |1⟩ state: (0, 0, -1) - south pole
- |+⟩ = (|0⟩ + |1⟩)/√2: (1, 0, 0) - +X axis
- |-⟩ = (|0⟩ - |1⟩)/√2: (-1, 0, 0) - -X axis
- |+i⟩ = (|0⟩ + i|1⟩)/√2: (0, 1, 0) - +Y axis
- |-i⟩ = (|0⟩ - i|1⟩)/√2: (0, -1, 0) - -Y axis

## HTML Artifact Contract

The `bloch_to_html()` function must produce:

1. **Self-contained HTML** - Can be opened directly in a browser
2. **3D visualization** - Interactive Bloch sphere rendering
3. **Point marker** - Visual indicator at coordinates (x, y, z)
4. **Coordinate display** - Text showing (x, y, z) values
5. **Title** - Configurable title for the visualization

**Recommended approach:**
- Use a lightweight 3D library (e.g., Three.js via CDN)
- Or use plotly.js for interactive 3D scatter plots
- Or use matplotlib's 3D plotting exported to HTML (via mpld3 or plotly backend)

**File format:**
- Single HTML file
- No external file dependencies (or CDN links only)
- Viewable offline after initial load
