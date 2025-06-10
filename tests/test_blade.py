import unittest
from rotor_toolkit.blade import Blade
from rotor_toolkit.airfoil import Airfoil

class TestBlade(unittest.TestCase):
    def test_init(self):
        af = Airfoil(name="TestAF")
        blade = Blade(name="TestBlade", airfoil_sections=[(af, 0.0), (af, 1.0)])
        self.assertEqual(blade.name, "TestBlade")
        self.assertEqual(len(blade.airfoil_sections), 2)

if __name__ == "__main__":
    unittest.main()
