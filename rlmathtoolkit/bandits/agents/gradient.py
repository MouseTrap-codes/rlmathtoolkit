import numpy as np
from rlmathtoolkit.bandits.agents.base import Agent
from rlmathtoolkit.bandits.policies import softmax_sample, softmax
from rlmathtoolkit.bandits.updates import gradient_pref_update


class GradientBanditAgent(Agent):
  def __init__(self, n, alpha, use_baseline=True):
    super().__init__(n)
    self.alpha = alpha
    self.h = np.zeros(n)
    self.pi = np.zeros(n)
    self.average_reward = 0.0
    self.use_baseline = use_baseline
    self.t = 0
  
  def select_action(self):
    # calculate probaiilities using softmax distribution
    self.pi = softmax(self.h)

    # select action from sampling from the softmax distribution
    return softmax_sample(self.pi)
  
  def update(self, action, reward):
    self.t += 1

    if self.use_baseline:
      # use incremental average update to update average reward 
      self.average_reward += (1 / self.t) * (reward - self.average_reward)
    else:
      self.average_reward = 0.0

    # update preferences
    gradient_pref_update(self.pi, self.h, self.average_reward, reward, self.alpha, A_t=action)