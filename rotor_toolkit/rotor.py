"""
Rotor class for assembling blades and hub into a full rotor.
"""
from .blade import Blade
from .hub import Hub

class Rotor:
    """
    Represents a rotor, composed of a hub and multiple blades.
    """
    def __init__(self, name, hub, blades):
        """
        Args:
            name (str): Name of the rotor.
            hub (Hub): Hub object.
            blades (list of Blade): List of Blade objects.
        """
        self.name = name
        self.hub = hub
        self.blades = blades

    def assemble(self):
        """
        Assemble the rotor geometry from hub and blades.
        """
        pass

    def plot(self):
        """
        Plot the rotor (3D visualization).
        """
        pass

    def export(self, filename, fmt="stl"):
        """
        Export rotor geometry to a file (STL, etc.).
        """
        pass
