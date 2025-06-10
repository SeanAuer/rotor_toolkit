# rotor_toolkit

`rotor_toolkit` is a Python-based design and modeling library for rotorcraft and turbomachinery components, with a focus on modular, parametric geometry generation. The project began as a way to generate and manipulate 2D airfoils but is growing into a pipeline for defining and exporting full 3D rotor or stator assemblies for UAS, turbomachinery, hydrodynamic vehicles, and helicopters.

The toolkit is designed to support exploratory design, optimization, and integration with external analysis and iteration tools. It emphasizes parametric modeling of airfoil and blade sections, with the ability to generate and output clean, structured geometry for downstream applications.

---

## Components

### `Airfoil`
- Supports NACA 4-digit and 5-digit generation
- Can load `.dat` or `.csv` airfoil files
- Allows user-defined coordinate lists
- Spline-based generation using B-spline control points
- Supports chord normalization, alignment, and plotting
- Returns airfoil characteristics dynamically

### `Blade` (coming soon)
- Accepts a list of airfoil sections and spanwise definitions
- Will support twist, chord, and sweep distributions
- Generates 3D blade surface for lofting or export
- Will include sectional visualization and thickness plotting

### `Hub` (planned)
- Represents central hub geometry
- Allows placement of blades with defined angular spacing
- Simple geometric representation for early design/export

### `Rotor`
- Combines blades + hub + orientation
- Can define stators or rotors
- Will support clean geometry export (STL, surface grid, etc.)
- Planned integration with meshing/export utilities

---

## Current Capabilities

- Generate clean, smooth 2D airfoils using standard definitions or parametric control
- Plot airfoils interactively (`matplotlib` and `plotly`)
- Evaluate geometric metrics like chord length
- Export to CSV for post-processing
- Align airfoils along chord for consistency
- Manipulate control points on spline airfoils

---

## Example (Basic)

```python
from rotor_toolkit.airfoil import Airfoil

# Generate a classic NACA 2412 4-digit airfoil
af = Airfoil(name="NACA2412", naca_code="2412", n_points=200)
af.plot(save=True)

# Create a spline NACA 2412 airfoil from control points
import numpy as np
control_pts = np.array([
    [0.0,  0.0],
    [0.1,  0.06],
    [0.3,  0.08],
    [0.6,  0.02],
    [0.9, -0.01],
    [1.0,  0.0]])
spline_af = Airfoil.from_spline("custom_spline", control_pts)
spline_af.plot()
```

---

## Philosophy

This project is intended to provide clarity, extensibility, and practicality for engineers who want direct control over airfoil and rotor geometry in a Python environment.

The goal is to support:
- Hands-on geometry scripting
- Rapid iteration
- Future plug-in compatibility for meshing and MDO workflows

---</file>