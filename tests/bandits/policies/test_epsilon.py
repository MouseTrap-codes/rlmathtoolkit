import numpy as np
from rlmathtoolkit.bandits.policies import e_greedy, greedy

# test greedy
def test_greedy_returns_max_index():
    q = [1, 2, 3, 5, 4]
    assert greedy(q) == 3

def test_greedy_returns_first_max_on_tie():
    q = [4.0, 10.0, 11.0, 11.0, 10.0, 9.0]
    assert greedy(q) == 2

#test e-greedy
def test_e_greedy_returns_greedy_when_epsilon_zero():
    np.random.seed(42)
    q_values = [1.0, 5.0, 2.0]
    epsilon = 0.0
    assert e_greedy(q_values, epsilon) == 1

def test_e_greedy_returns_random_when_epsilon_one():
    np.random.seed(42)
    q_values = [10.0, -5.0, 3.0]
    epsilon = 1.0
    chosen = e_greedy(q_values, epsilon)
    assert chosen in range(len(q_values))  # must be a valid index


def test_e_greedy_approx_behavior_over_many_trials():
    np.random.seed(0)
    q_values = [1.0, 2.0, 3.0]
    epsilon = 0.2
    counts = [0, 0, 0] 
    N = 10000
    for _ in range(N):
        a = e_greedy(q_values, epsilon)
        counts[a] += 1

    greedy_index = greedy(q_values)
    greedy_pct = counts[greedy_index] / N

    # P(greedy) = (1 - epsilon) + epsilon * 1/n

    # expected greedy rate ≈ 0.8667
    assert 0.85 < greedy_pct < 0.88



