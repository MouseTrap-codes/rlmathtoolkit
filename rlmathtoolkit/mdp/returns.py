import numpy as np

# all discounted returns
def discounted_returns(rewards, gamma):
    k = len(rewards)
    G = np.zeros(k)
    G_t = 0
    for t in reversed(range(rewards)):
        G_t = rewards[t] + gamma * G_t
        G[t] = G_t
    return G





