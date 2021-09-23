import numpy as np

from cca.cca import CCA


def test_CCA():
    Y = [np.zeros((10, 10)), np.zeros((10, 10))]

    model = CCA(Y, 5)
