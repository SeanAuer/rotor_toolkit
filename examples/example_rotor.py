"""
Example script for testing parametric rotor build using rotor_toolkit.
"""
import rotor_toolkit as rt
from rotor_toolkit.airfoil import Airfoil
from rotor_toolkit.blade import Blade
from rotor_toolkit.hub import Hub
from rotor_toolkit.rotor import Rotor 

def main():
    # Create airfoils using NACA definitions
    af1 = Airfoil(name="NACA0012", naca_code="0012", n_points=200)
    af2 = Airfoil(name="NACA4412", naca_code="4412", n_points=200)

    # Create a blade from airfoil sections
    blade = Blade(
        name="Blade1",
        airfoil_sections=[(af1, 0.0), (af2, 1.0)],
        span=1.0,
        chord_distribution=[1.0, 0.8],  # optional test input
        twist_distribution=[0.0, 15.0]  # optional test input
    )

    # Create hub
    hub = Hub(name="Hub1", n_blades=3, radius=0.1)

    # Combine into a rotor
    rotor = Rotor(name="Rotor1", hub=hub, blades=[blade])

    # Plot or inspect rotor properties
    print(rotor)
    # Placeholder for visualization
    # rt.plot(rotor) or rotor.plot() if method is defined

if __name__ == "__main__":
    main()
