import unittest
from rotor_toolkit.hub import Hub

class TestHub(unittest.TestCase):
    def test_blade_positions(self):
        hub = Hub(name="Hub1", n_blades=3)
        positions = hub.get_blade_positions()
        self.assertEqual(len(positions), 3)

if __name__ == "__main__":
    unittest.main()
