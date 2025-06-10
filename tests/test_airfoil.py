import unittest
from rotor_toolkit.airfoil import Airfoil
import numpy as np

class TestAirfoil(unittest.TestCase):
    def test_init(self):
        af = Airfoil(name="TestAF")
        self.assertEqual(af.name, "TestAF")

    def test_generate_coordinates(self):
        af = Airfoil(name="TestAF", equation=lambda n: np.zeros((n,2)))
        coords = af.generate_coordinates(10)
        self.assertEqual(coords.shape, (10,2))

    def test_naca4_generation(self):
        af = Airfoil(name="NACA2412", naca_code="2412", n_points=100)
        self.assertEqual(af.coordinates.shape[1], 2)
        self.assertTrue(np.allclose(af.coordinates[0], af.coordinates[-1]))

    def test_naca5_generation(self):
        af = Airfoil(name="NACA23015", naca_code="23015", n_points=100)
        self.assertEqual(af.coordinates.shape[1], 2)
        self.assertTrue(np.allclose(af.coordinates[0], af.coordinates[-1]))

    def test_naca6_generation(self):
        af = Airfoil(name="NACA63A415", naca_code="63A415", n_points=100)
        self.assertEqual(af.coordinates.shape[1], 2)
        self.assertTrue(np.allclose(af.coordinates[0], af.coordinates[-1]))

if __name__ == "__main__":
    unittest.main()
