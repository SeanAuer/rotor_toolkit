"""
Example script for building and visualizing a rotor using rotor_toolkit.
"""
import rotor_toolkit as rt

def main():
    # Example: create airfoils, blade, hub, and rotor
    af1 = rt.Airfoil(name="NACA0012")
    af2 = rt.Airfoil(name="NACA4412")
    blade = rt.Blade(name="Blade1", airfoil_sections=[(af1, 0.0), (af2, 1.0)])
    hub = rt.Hub(name="Hub1", n_blades=1, radius=0.1)
    rotor = rt.Rotor(name="Rotor1", hub=hub, blades=[blade])
    # Visualize (placeholder)
    rotor.plot()

if __name__ == "__main__":
    main()
