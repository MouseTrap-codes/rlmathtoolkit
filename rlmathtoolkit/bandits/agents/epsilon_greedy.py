import numpy as np
from rlmathtoolkit.bandits.agents.base import Agent
from rlmathtoolkit.bandits.policies import e_greedy
from rlmathtoolkit.bandits.updates import sample_average_method, exponential_recency_weighted_average_method


class EpsilonGreedyAgent(Agent):
  def __init__(self, n, epsilon, alpha=None, initial_estimates=0.0):
    super().__init__(n)
    self.epsilon = epsilon
    self.q = np.full(n, initial_estimates) 
    self.n_t = np.zeros(n)
    self.alpha = alpha # None -> use sample average
    
  
  def select_action(self):
    return e_greedy(self.q, self.epsilon)
  
  def update(self, action, reward):
    self.n_t[action] += 1

    if self.alpha is None:
        self.q[action] = sample_average_method(self.q[action], reward, self.n_t[action])
    else:
        self.q[action] = exponential_recency_weighted_average_method(self.q[action], reward, self.n_t[action], alpha=self.alpha)
  
  
