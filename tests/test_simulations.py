import numpy as np

from cca.simulations import Simulation


def test_simulation_reproducibility():
    s0 = Simulation(341)
    s1 = Simulation(341)

    assert np.all(s0.Y[0] == s1.Y[0])
