import numpy as np

# all discounted returns
def discounted_returns(rewards, gamma):
    k = len(rewards)
    G = np.zeros(k)
    for t in reversed(k):
        G_t = rewards[t] + gamma * G_t
        G[t] = G_t
    return G





