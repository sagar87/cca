from cca.cca import CCA
import numpy as np


def test_CCA():
    Y = [np.zeros((10, 10)), np.zeros((10, 10))]
    
    model = CCA(Y, 5)
    