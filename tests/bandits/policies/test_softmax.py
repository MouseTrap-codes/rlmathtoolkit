import numpy as np
from rlmathtoolkit.bandits.policies import softmax

def test_softmax_normalization():
    h = [1.0, 2.0, 3.0]
    pi = softmax(h)
    assert np.isclose(np.sum(pi), 1.0)

def test_softmax_shape_and_range():
    h = [1.0, 2.0, 3.0, 5.0, 9.0]
    pi = softmax(h)
    assert len(pi) == len(h)
    assert np.all(pi >= 0) and np.all(pi <= 1)

def test_softmax_uniform_output():
    h = [1.0, 1.0, 1.0, 1.0]
    pi = softmax(h)
    expected = [.25, .25, .25, .25]
    assert np.allclose(pi, expected)

def test_softmax_large_values():
    h = [10000, 23300, 2010]
    pi = softmax(h)
    assert np.isclose(np.sum(pi), 1.0)

def test_softmax_sharp_peak():
    h = [1, 10000, 2]
    pi = softmax(h)
    assert np.argmax(pi) == 1
    assert pi[1] > 0.99