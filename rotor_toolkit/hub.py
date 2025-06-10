"""
Hub class for rotor hub geometry and blade mounting.
"""
class Hub:
    """
    Represents a rotor hub, which positions blades azimuthally.
    """
    def __init__(self, name, n_blades, radius=0.1):
        """
        Args:
            name (str): Name of the hub.
            n_blades (int): Number of blades.
            radius (float): Hub radius (default 0.1).
        """
        self.name = name
        self.n_blades = n_blades
        self.radius = radius

    def get_blade_positions(self):
        """
        Return azimuthal positions for each blade.
        """
        import numpy as np
        return [2 * np.pi * i / self.n_blades for i in range(self.n_blades)]
