import numpy as np
from bandits.agents.base import Agent
from bandits.policies import ucb_select
from bandits.updates import sample_average_method, exponential_recency_weighted_average_method


class UCBAgent(Agent):
  def __init__(self, n, c, alpha: float | None = None):
    super().__init__(n)
    self.c = c
    self.q = np.zeros(n)
    self.n_t = np.zeros(n)
    self.alpha = alpha # None -> sample average
    
  
  def select_action(self):
    return ucb_select(self.q, self.n_t, self.c)
  
  def update(self, action, reward):
    self.n_t[action] += 1
    if self.alpha is None:
        self.q[action] = sample_average_method(self.q[action], reward, self.n_t[action])
    else:
        self.q[action] = exponential_recency_weighted_average_method(self.q[action], reward, self.n_t[action])
  
  
