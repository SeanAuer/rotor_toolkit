"""
Example script to test parametric airfoil creation and manipulation from a NACA 5-series airfoil.
This recasts a NACA 5-series to a B-spline representation and displays control points and properties.
"""

from rotor_toolkit.airfoil import Airfoil, plot_airfoils
import numpy as np

def main():
    # Step 1: Generate NACA 5-digit airfoil
    original_af = Airfoil(name="NACA23012", naca_code="23012", n_points=100)
    print("Original NACA 23012 Airfoil:")
    original_af.report()

    # Step 2: Convert to spline-based airfoil
    spline_af = original_af.to_spline(n_points=200, degree=3)
    print("\nConverted to Spline Airfoil:")
    spline_af.report()

    # Step 3: Simplest symmetric airfoil from control points and tangencies
    # Only LE and TE, no intermediate points
    option = 2
    if option == 1:
        upper_points = np.empty((0, 2))  # No intermediate points
        lower_points = np.empty((0, 2))
        upper_angles = []  # No intermediate tangencies
        lower_angles = []
        upper_trailing_edge_angle = 7.5  # Flat at TE
        lower_trailing_edge_angle = -2.5
    elif option == 2:
        upper_points = np.array([0.2, -0.05])  # No intermediate points
        lower_points = np.empty((0, 2))
        upper_angles = [0]  # No intermediate tangencies
        lower_angles = []
        upper_trailing_edge_angle = 10.0  # Flat at TE
        lower_trailing_edge_angle = 5.0


    param_af = Airfoil.from_control_points(
        name="SimpleSymmetricAirfoil",
        upper_surface_control_points=upper_points,
        lower_surface_control_points=lower_points,
        upper_surface_angles=upper_angles,
        lower_surface_angles=lower_angles,
        upper_trailing_edge_angle=upper_trailing_edge_angle,
        lower_trailing_edge_angle=lower_trailing_edge_angle,
        n_points=100
    )
    print("\nSimplest Symmetric Airfoil from Control Points:")
    param_af.report()
    print("Upper Control Points:\n", param_af.upper_surface_control_points)
    print("Lower Control Points:\n", param_af.lower_surface_control_points)

    # Step 4: (Optional) File-based airfoil if you have a .dat or .csv file
    # Uncomment and set your file path if you want to test file import
    # file_af = Airfoil(name="FileAirfoil", file="path/to/airfoil.dat")
    # print("\nFile-based Airfoil:")
    # file_af.report()

    # Step 5: Plot all available airfoils for comparison
    airfoils_to_plot = [original_af, spline_af, param_af]
    # If file_af is defined, add it:
    # airfoils_to_plot.append(file_af)
    plot_airfoils(airfoils_to_plot)

if __name__ == "__main__":
    main()