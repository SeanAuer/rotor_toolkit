# rotor_toolkit

`rotor_toolkit` is a Python-based design and modeling library for rotorcraft and turbomachinery components, with a focus on modular, parametric geometry generation. The project began as a way to generate and manipulate 2D airfoils but is growing into a pipeline for defining and exporting full 3D rotor or stator assemblies for UAS, turbomachinery, hydrodynamic vehicles, and helicopters.

---

## Requirements

- Python 3.8+
- [`spline_toolkit`](https://github.com/SeanAuer/spline_toolkit) — required for spline-based geometry definitions

Make sure to install `spline_toolkit` before using the spline-based airfoil generation features.

---

## Components

### `Airfoil`
- Supports NACA 4-digit and 5-digit airfoil generation
- Accepts `.dat` or `.csv` airfoil files
- Allows fully custom coordinate input
- Parametric spline generation using control points (via `spline_toolkit`)
- Chord normalization, alignment to x-axis, and centerline enforcement
- Interactive plotting with support for matplotlib/plotly
- Dynamic evaluation of airfoil characteristics

### `Blade` (in work)
- Accepts a list of airfoil sections and spanwise definitions
- Will support twist, chord, and sweep distributions
- Generates 3D blade surface for lofting or export
- Will include sectional visualization and thickness plotting

### `Hub` (planned)
- Represents central hub geometry
- Allows placement of blades with defined angular spacing
- Simple geometric representation for early design/export

### `Rotor` (planned)
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
from rotor_toolkit import Airfoil

# Generate a classic NACA 2412 4-digit airfoil
af = Airfoil(name="NACA2412", naca_code="2412", n_points=200)
af.plot()
```

---

## Philosophy

`rotor_toolkit` aims to enable parametric rotor and airfoil modeling in a streamlined, Python-based environment.

Goals:
- Script-based geometry creation and evaluation
- Quick iteration for design exploration
- Future integration with meshing, optimization, and export pipelines

---

## License

This project is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.