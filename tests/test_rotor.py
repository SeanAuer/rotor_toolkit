import unittest
from rotor_toolkit.rotor import Rotor
from rotor_toolkit.hub import Hub
from rotor_toolkit.blade import Blade
from rotor_toolkit.airfoil import Airfoil

class TestRotor(unittest.TestCase):
    def test_init(self):
        af = Airfoil(name="TestAF")
        blade = Blade(name="TestBlade", airfoil_sections=[(af, 0.0), (af, 1.0)])
        hub = Hub(name="Hub1", n_blades=1)
        rotor = Rotor(name="Rotor1", hub=hub, blades=[blade])
        self.assertEqual(rotor.name, "Rotor1")
        self.assertEqual(len(rotor.blades), 1)

if __name__ == "__main__":
    unittest.main()
