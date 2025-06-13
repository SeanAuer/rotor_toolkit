"""
Airfoil class for 2D airfoil geometry definition, manipulation, and export.
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy.interpolate import BSpline

class Airfoil:
    """
    Represents a 2D airfoil shape, defined by coordinates or parametric equations.
    """
    def __init__(self, name, coordinates=None, equation=None, file=None, naca_code=None, n_points=100):
        """
        Initialize an Airfoil.
        Args:
            name (str): Name of the airfoil.
            coordinates (np.ndarray): Nx2 array of (x, y) points.
            equation (callable): Function to generate (x, y) points.
            file (str): Filename to read coordinates from.
            naca_code (str): NACA 4 or 5-digit code to generate airfoil coordinates.
            n_points (int): Number of points to generate along the airfoil.
        """
        self.name = name
        self.coordinates = coordinates
        self.equation = equation
        self.file = file
        self.naca_code = naca_code
        self.n_points = n_points
        if naca_code is not None:
            if len(naca_code) == 4 and naca_code.isdigit():
                self.coordinates = self._generate_naca4_coordinates(naca_code, n_points)
            elif len(naca_code) == 5 and naca_code.isdigit():
                self.coordinates = self._generate_naca5_coordinates(naca_code, n_points)
            else:
                raise ValueError(f"Unrecognized NACA code: {naca_code}")
        elif file is not None:
            self.coordinates = self._read_coordinates_from_file(file)
        elif equation is not None:
            self.coordinates = equation(n_points)
        # else: coordinates must be provided directly
        if self.coordinates is not None:
            self._ensure_closed()

    @classmethod
    def from_spline(cls, upper_surface, lower_surface, name=None, n_points=100):
        """
        Create an Airfoil from upper and lower spline/curve objects.
        The objects must implement a .sample(n) method returning an (n, d) ndarray.
        Args:
            upper_surface: Curve or spline object with a .sample(n) method (upper surface).
            lower_surface: Curve or spline object with a .sample(n) method (lower surface).
            name (str): Name of the airfoil.
            n_points (int): Number of points to sample per surface.
        Returns:
            Airfoil: Constructed airfoil instance.
        """
        import numpy as np

        upper_coords = upper_surface.sample(n_points)
        lower_coords = lower_surface.sample(n_points)

        # Check LE and TE alignment
        if not np.allclose(upper_coords[0], lower_coords[0], atol=1e-6):
            raise ValueError("Leading edges do not align between upper and lower surfaces.")
        if not np.allclose(upper_coords[-1], lower_coords[-1], atol=1e-6):
            raise ValueError("Trailing edges do not align between upper and lower surfaces.")

        # Combine into full airfoil coordinates
        coords = np.vstack((upper_coords, lower_coords[::-1]))

        # Ensure closed loop
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        airfoil = cls(name=name, coordinates=coords, n_points=len(coords))
        airfoil.upper_surface = upper_surface
        airfoil.lower_surface = lower_surface
        return airfoil

    @classmethod
    def from_curves(cls, upper_curve, lower_curve, name=None, n_points=100, curve_type='hermite'):
        """
        Create an Airfoil from upper and lower curve objects (e.g., QuinticHermiteCurve).
        Args:
            upper_curve: Curve object (must have .sample(n) method) for upper surface.
            lower_curve: Curve object (must have .sample(n) method) for lower surface.
            name (str): Name of the airfoil.
            n_points (int): Number of points to sample per surface.
            curve_type (str): 'hermite' or 'bezier' (future-proof).
        Returns:
            Airfoil: Constructed airfoil instance.
        """
        import numpy as np
        # Helper to ensure we have a Curve object, not just a Spline
        def _curve_from_input(obj, curve_type):
            if curve_type == 'hermite':
                from spline_toolkit import QuinticHermiteCurve, QuinticHermiteSpline
                if isinstance(obj, QuinticHermiteCurve):
                    return obj
                elif isinstance(obj, QuinticHermiteSpline):
                    # Convert Spline to Curve
                    pts = obj.get_control_points()
                    tans = obj.get_tangents()
                    crvs = obj.get_curvatures()
                    return QuinticHermiteCurve.from_controls(pts, tans, crvs)
                else:
                    raise TypeError("Input must be a QuinticHermiteCurve or QuinticHermiteSpline for 'hermite' type.")
            elif curve_type == 'bezier':
                # Placeholder for future BezierCurve support
                raise NotImplementedError("Bezier curve support not yet implemented.")
            else:
                raise ValueError(f"Unknown curve_type: {curve_type}")

        upper = _curve_from_input(upper_curve, curve_type)
        lower = _curve_from_input(lower_curve, curve_type)
        upper_coords = upper.sample(n_points)
        lower_coords = lower.sample(n_points)
        # Check LE and TE alignment
        if not np.allclose(upper_coords[0], lower_coords[0], atol=1e-6):
            raise ValueError("Leading edges do not align between upper and lower surfaces.")
        if not np.allclose(upper_coords[-1], lower_coords[-1], atol=1e-6):
            raise ValueError("Trailing edges do not align between upper and lower surfaces.")
        # Combine into full airfoil coordinates
        coords = np.vstack((upper_coords, lower_coords[::-1]))
        # Ensure closed loop
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])
        # Create Airfoil and align to x-axis
        airfoil = cls(name=name, coordinates=coords, n_points=len(coords))
        airfoil._align_to_x_axis()
        airfoil.upper_curve = upper
        airfoil.lower_curve = lower
        return airfoil

    @property
    def upper_surface_control_points(self):
        """
        Return upper surface control points if defined (used for parametric design).
        """
        return getattr(self, '_upper_surface_control_points', None)

    @upper_surface_control_points.setter
    def upper_surface_control_points(self, points):
        """
        Set upper surface control points.
        """
        self._upper_surface_control_points = np.array(points)

    @property
    def lower_surface_control_points(self):
        """
        Return lower surface control points if defined (used for parametric design).
        """
        return getattr(self, '_lower_surface_control_points', None)

    @lower_surface_control_points.setter
    def lower_surface_control_points(self, points):
        """
        Set lower surface control points.
        """
        self._lower_surface_control_points = np.array(points)

    @classmethod
    def from_control_points(
        cls,
        name,
        upper_surface_control_points,
        lower_surface_control_points,
        upper_surface_angles,
        lower_surface_angles,
        upper_trailing_edge_angle,
        lower_trailing_edge_angle,
        n_points=100,
        curve_type="hermite"
    ):
        """
        Create an Airfoil using control points and tangency angles at each point.
        Args:
            name (str): Airfoil name.
            upper_surface_control_points (list of [x, y, ...]): Upper surface points.
            lower_surface_control_points (list of [x, y, ...]): Lower surface points.
            upper_surface_angles (list of float): Tangent angles (deg) at each upper point.
            lower_surface_angles (list of float): Tangent angles (deg) at each lower point.
            upper_trailing_edge_angle (float): TE angle (deg) upper surface.
            lower_trailing_edge_angle (float): TE angle (deg) lower surface.
            n_points (int): Points to sample per surface.
            curve_type (str): Type of curve to use ("hermite", etc.).
        Returns:
            Airfoil: Airfoil object with interpolated coordinates.
        """

        def angle_to_vector(angle_deg):
            angle_rad = np.deg2rad(angle_deg)
            return np.array([np.cos(angle_rad), np.sin(angle_rad)])

        def process_surface(points, angles, is_upper, trailing_angle):
            points = np.array(points)
            angles = list(angles)
            if points.shape[0] == 0:
                points = np.array([[0.0, 0.0], [1.0, 0.0]])
                angles = [270.0 if is_upper else 90.0, trailing_angle]
                return points, angles
            if not np.allclose(points[0], [0.0, 0.0]):
                points = np.vstack([[0.0, 0.0], points])
                default_le_angle = 270.0 if is_upper else 90.0
                angles = [default_le_angle] + angles
            if not np.allclose(points[-1], [1.0, 0.0]):
                points = np.vstack([points, [1.0, 0.0]])
                angles = angles + [trailing_angle]
            return points, angles

        upper_cp, upper_angles = process_surface(
            upper_surface_control_points, upper_surface_angles, is_upper=True, trailing_angle=upper_trailing_edge_angle
        )
        lower_cp, lower_angles = process_surface(
            lower_surface_control_points, lower_surface_angles, is_upper=False, trailing_angle=lower_trailing_edge_angle
        )

        upper_tangents = np.array([angle_to_vector(a) for a in upper_angles])
        lower_tangents = np.array([angle_to_vector(a) for a in lower_angles])

        if curve_type.lower() == "hermite":
            # Pass full arrays for arbitrary dimensionality; include optional curvature support.
            upper_curvatures = getattr(cls, '_upper_surface_curvatures', None)
            lower_curvatures = getattr(cls, '_lower_surface_curvatures', None)
            upper_curve = QuinticHermiteCurve.from_controls(upper_cp, upper_tangents, upper_curvatures)
            lower_curve = QuinticHermiteCurve.from_controls(lower_cp, lower_tangents, lower_curvatures)
        else:
            raise NotImplementedError(f"Unsupported curve_type: {curve_type}")

        upper_coords = upper_curve.sample(n_points)
        lower_coords = lower_curve.sample(n_points)

        coords = np.vstack((upper_coords, lower_coords[::-1]))
        airfoil = cls(name=name, coordinates=coords, n_points=len(coords))
        airfoil._upper_surface_control_points = upper_cp
        airfoil._lower_surface_control_points = lower_cp
        return airfoil

    def _generate_naca4_coordinates(self, code, n_points):
        """
        Generate coordinates for a NACA 4-digit airfoil using cosine spacing for better clustering near high-curvature regions.
        Also sets upper_surface and lower_surface attributes for consistency with spline-based airfoils.
        """
        m = int(code[0]) / 100.0
        p = int(code[1]) / 10.0
        t = int(code[2:]) / 100.0
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1 - np.cos(beta))
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        for i, xi in enumerate(x):
            if xi < p and p != 0:
                yc[i] = m / p**2 * (2 * p * xi - xi**2)
                dyc_dx[i] = 2 * m / p**2 * (p - xi)
            elif p != 0:
                yc[i] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * xi - xi**2)
                dyc_dx[i] = 2 * m / (1 - p)**2 * (p - xi)
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        x_coords = np.concatenate([xu[::-1], xl[1:]])
        y_coords = np.concatenate([yu[::-1], yl[1:]])
        coords = np.stack([x_coords, y_coords], axis=1)
        # Set upper_surface and lower_surface as simple objects with .sample(n)
        class NacaSurface:
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def sample(self, n):
                idx = np.linspace(0, len(self.x)-1, n).astype(int)
                return np.stack([self.x[idx], self.y[idx]], axis=1)
        self.upper_surface = NacaSurface(xu[::-1], yu[::-1])
        self.lower_surface = NacaSurface(xl[1:], yl[1:])
        return coords

    def _generate_naca5_coordinates(self, code, n_points):
        """
        Generate coordinates for a NACA 5-digit airfoil using cosine spacing.
        Also sets upper_surface and lower_surface attributes for consistency with spline-based airfoils.
        """
        cld = int(code[0]) * 0.15
        p = int(code[1:3]) / 20.0
        t = int(code[3:]) / 100.0
        reflex = int(code[3]) >= 5 if len(code) == 5 else False
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1 - np.cos(beta))
        m = cld / 0.3
        if reflex:
            k1 = 0.006
        else:
            k1 = 0.058
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        for i, xi in enumerate(x):
            if xi < p:
                yc[i] = k1/6 * (xi**3 - 3*p*xi**2 + p**2*(3-xi))
                dyc_dx[i] = k1/6 * (3*xi**2 - 6*p*xi + p**2*(-1))
            else:
                yc[i] = k1/6 * p**3 * (1-xi)
                dyc_dx[i] = k1/6 * p**3 * (-1)
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        x_coords = np.concatenate([xu[::-1], xl[1:]])
        y_coords = np.concatenate([yu[::-1], yl[1:]])
        coords = np.stack([x_coords, y_coords], axis=1)
        class NacaSurface:
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def sample(self, n):
                idx = np.linspace(0, len(self.x)-1, n).astype(int)
                return np.stack([self.x[idx], self.y[idx]], axis=1)
        self.upper_surface = NacaSurface(xu[::-1], yu[::-1])
        self.lower_surface = NacaSurface(xl[1:], yl[1:])
        return coords

    @staticmethod
    def _read_coordinates_from_file(file):
        """
        Read airfoil coordinates from a file (CSV or DAT).
        """
        try:
            coords = np.loadtxt(file, delimiter=None)
            return coords
        except Exception:
            raise ValueError(f"Could not read airfoil coordinates from {file}")

    def _ensure_closed(self):
        """
        Ensure the airfoil coordinates form a closed loop (start and end points match).
        """
        if not np.allclose(self.coordinates[0], self.coordinates[-1]):
            self.coordinates = np.vstack([self.coordinates, self.coordinates[0]])

    def generate_coordinates(self, n_points=100):
        """
        Generate airfoil coordinates from equation if not provided.
        """
        if self.equation is not None:
            self.coordinates = self.equation(n_points)
        return self.coordinates

    def plot(self, save=False, n_points=200):
        """
        Plot the airfoil shape using matplotlib with a cinematic German-inspired style.
        Args:
            save (bool): If True, save the plot to a PNG file named after the airfoil.
            n_points (int): Number of points to sample per surface (default 200). If upper/lower surfaces exist, resample them.
        """
        import matplotlib.pyplot as plt
        coords = None
        # If the airfoil was built from upper_surface/lower_surface, resample for smoothness
        if hasattr(self, 'upper_surface') and hasattr(self, 'lower_surface'):
            print('good')
            try:
                upper_coords = self.upper_surface.sample(n_points//2)
                lower_coords = self.lower_surface.sample(n_points//2)
            except Exception:
                upper_coords = self.upper_surface.samples[:n_points//2]
                lower_coords = self.lower_surface.samples[:n_points//2]
            coords = np.vstack((upper_coords, lower_coords[::-1]))
        else:
            print('no good')
            coords = self.coordinates if self.coordinates is not None else self.generate_coordinates()
        fig, ax = plt.subplots(figsize=(10, 4))  # Cinematic aspect ratio
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.plot(coords[:,0], coords[:,1], color='red', linewidth=2)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', colors='white', which='both')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.set_title(f"Airfoil: {self.name}", color='white', fontsize=16, pad=20)
        plt.tight_layout()
        plt.grid(True)
        if save:
            plt.savefig(f"{self.name}.png", dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()

    def resample(self, n_points=300):
        """
        Re-sample the upper and lower splines using curvature-based distribution
        with a total of `n_points` per surface.
        Typically used for higher-fidelity plotting or modeling.
        """
        if hasattr(self, "upper_curve") and hasattr(self, "lower_curve"):
            self.upper_points = self.upper_curve.sample(n_points=n_points)
            self.lower_points = self.lower_curve.sample(n_points=n_points)
        else:
            raise AttributeError("Airfoil does not have upper_curve or lower_curve attributes for resampling.")

    def export(self, filename, fmt="csv"):
        """
        Export airfoil coordinates to a file (csv, etc.).
        """
        coords = self.coordinates if self.coordinates is not None else self.generate_coordinates()
        if fmt == "csv":
            np.savetxt(filename, coords, delimiter=",", header="x,y", comments="")
        # Add more formats as needed

    def _align_to_x_axis(self):
        """
        Shift and rotate airfoil so that the leading edge is at (0,0),
        the trailing edge is at (1,0), and the chord is horizontal and points right.
        """
        coords = self.coordinates.copy()
        le_idx = np.argmin(coords[:, 0])
        le = coords[le_idx]
        # Find trailing edge: farthest point from leading edge
        dists = np.linalg.norm(coords - le, axis=1)
        te_idx = np.argmax(dists)
        te = coords[te_idx]
        # Translate so leading edge is at (0,0)
        coords = coords - le
        # Compute angle to x-axis
        chord_vec = te - le
        angle = np.arctan2(chord_vec[1], chord_vec[0])
        # Rotate so chord is on x-axis
        rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        coords = coords @ rot.T
        # Scale so trailing edge is at (1,0)
        te_new = coords[te_idx]
        scale = 1.0 / np.linalg.norm(te_new)
        coords = coords * scale
        # If trailing edge is to the left of leading edge, flip horizontally
        if coords[te_idx, 0] < 0:
            coords[:, 0] = -coords[:, 0]
        self.coordinates = coords

    def _chord_length(self):
        le_idx = np.argmin(self.coordinates[:, 0])
        le = self.coordinates[le_idx]
        dists = np.linalg.norm(self.coordinates - le, axis=1)
        te_idx = np.argmax(dists)
        te = self.coordinates[te_idx]
        return np.linalg.norm(te - le)

    @property
    def chord(self):
        """
        Get the chord length of the airfoil.
        Returns:
            float: Chord length.
        """
        return self._chord_length()

    def __repr__(self):
        return f"Airfoil(name='{self.name}', n_points={len(self.coordinates)}, chord={self._chord_length():.3f})"

    def to_spline(self, n_points=100, degree=3):
        """
        Fit a B-spline to the current airfoil and return a new Airfoil object with n_points.
        Args:
            n_points (int): Number of points to sample on the spline.
            degree (int): Degree of the B-spline.
        Returns:
            Airfoil: New Airfoil object with spline-generated coordinates.
        """
        from scipy.interpolate import splprep, splev
        coords = self.coordinates
        # Remove duplicate last point if closed
        if np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        tck, _ = splprep([coords[:,0], coords[:,1]], s=0, k=degree, per=True)
        t_vals = np.linspace(0, 1, n_points)
        x_vals, y_vals = splev(t_vals, tck)
        spline_coords = np.stack([x_vals, y_vals], axis=1)
        return Airfoil(name=f"{self.name}_spline", coordinates=spline_coords, n_points=n_points)

    def report(self):
        """
        Print a summary of airfoil properties: name, chord, max thickness, max camber, number of points.
        """
        coords = self.coordinates
        chord = self._chord_length()
        # Max thickness (distance between upper and lower surface at same x)
        x_vals = coords[:,0]
        y_vals = coords[:,1]
        # For NACA, upper surface is first half, lower is second half
        n = len(coords)//2
        if n > 1:
            thickness = np.max(np.abs(y_vals[:n] - y_vals[-1:-n-1:-1]))
        else:
            thickness = np.nan
        camber = np.max(np.abs(y_vals))
        print(f"Airfoil: {self.name}")
        print(f"  Chord length: {chord:.4f}")
        print(f"  Max thickness: {thickness:.4f}")
        print(f"  Max camber: {camber:.4f}")
        print(f"  Number of points: {len(coords)}")


def plot_airfoils_plotly(airfoil_list):
    """
    Plot multiple Airfoil objects in a single interactive plotly window.
    Each airfoil can be toggled on/off by clicking its legend entry.
    Args:
        airfoil_list (list): List of Airfoil objects.
    """
    fig = go.Figure()
    for af in airfoil_list:
        coords = af.coordinates if af.coordinates is not None else af.generate_coordinates()
        fig.add_trace(go.Scatter(
            x=coords[:,0],
            y=coords[:,1],
            mode='lines',
            name=af.name,
            line=dict(width=3)
        ))
    fig.update_layout(
        title="Airfoil Comparison (Plotly)",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False, color='white'),
        yaxis=dict(showgrid=False, zeroline=False, color='white'),
        legend=dict(font=dict(color='white')),
        width=900,
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # Keep geometry 1:1
    fig.show()

def plot_airfoils(airfoil_list):
    """
    Plot multiple Airfoil objects in a single matplotlib window with a legend.
    Each airfoil is aligned to the x-axis for direct comparison.
    Legend entries can be clicked to toggle visibility.
    """
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4))
    lines = []
    for af in airfoil_list:
        af._align_to_x_axis()
        coords = af.coordinates
        line, = ax.plot(coords[:,0], coords[:,1], label=af.name, linewidth=2)
        lines.append(line)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Airfoil Comparison (Aligned Leading/Trailing Edges)')
    leg = ax.legend(loc='upper right', fancybox=True, framealpha=0.5)
    lined = {leg_line: orig_line for leg_line, orig_line in zip(leg.get_lines(), lines)}
    def on_pick(event):
        leg_line = event.artist
        orig_line = lined[leg_line]
        visible = not orig_line.get_visible()
        orig_line.set_visible(visible)
        leg_line.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()
    for leg_line in leg.get_lines():
        leg_line.set_picker(True)
    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()