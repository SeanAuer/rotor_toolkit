"""
Example: Generating an Airfoil from Quintic Hermite Splines

This script demonstrates how to construct an airfoil using spline_toolkit's
QuinticHermiteCurve for the upper and lower surfaces. The airfoil is then
assembled using rotor_toolkit's Airfoil class, with automatic closure and plotting.
"""

from spline_toolkit import QuinticHermiteSegment, QuinticHermiteSpline
from rotor_toolkit import Airfoil
import numpy as np
import math
import matplotlib.pyplot as plt

# -------------------------
# Define Upper Surface
# -------------------------

# Control Points
p0 = np.array([0.0, 0.0])  # Leading edge
p1 = np.array([1.0, 0.0])  # Trailing edge

# Tangents 
t0 = [0.0, 1.0]  
t1 = [np.cos(math.radians(-8.0)), np.sin(math.radians(-8.0))]

# Curvatures (2nd derivatives of position)
c0 = np.array([15.0, -4.87])
c1 = np.array([0.0, 0.0])

# Create upper surface spline curve
upper_surface = QuinticHermiteSpline.from_controls(
    [p0, p1],
    [t0, t1],
    [c0, c1]
)

# -------------------------
# Define Lower Surface
# -------------------------

# Control Points (same as upper to ensure closure)
p2 = np.array([0.0, 0.0])  # Leading edge
p3 = np.array([1.0, 0.0])  # Trailing edge

#
# Tangents (set LE tangent to 90° down, TE tangent remains at +10°)
t2 = np.array([np.cos(math.radians(-90.0)), np.sin(math.radians(-90.0))])  
t3 = np.array([np.cos(math.radians(5.0)), np.sin(math.radians(5.0))])

# Curvatures
# Increased curvature at LE to simulate camber
c2 = np.array([5.0, 4.0])
c3 = np.array([0.0, 0.0])

# Create lower surface spline curve
lower_surface = QuinticHermiteSpline.from_controls(
    [p2, p3],
    [t2, t3],
    [c2, c3]
)

# -------------------------
# Create Airfoil and Plot
# -------------------------

# Assemble airfoil from spline curves
airfoil = Airfoil.from_spline(
    upper_surface=upper_surface,
    lower_surface=lower_surface,
    name="Spline Airfoil"
)
print(airfoil.upper_surface)
# Plot airfoil shape
airfoil.plot(n_points=1000, save=True) 