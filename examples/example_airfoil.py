from rotor_toolkit.airfoil import Airfoil

af = Airfoil(name="NACA2412", naca_code="2412")
af_spline = af.to_spline(n_points=200)
af.report()
af.plot(save=False)