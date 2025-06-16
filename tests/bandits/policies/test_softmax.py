import numpy as np
from rlmathtoolkit.bandits.policies import softmax, softmax_sample

def test_softmax_normalization():
    h = [1.0, 2.0, 3.0]
    pi = softmax(h)
    assert np.sum(pi) == 1