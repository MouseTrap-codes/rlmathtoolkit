import numpy as np

# greedy -> select action with greatest estimated value
def greedy(q_t_values):
  return np.argmax(q_t_values)

# e-greedy -> with probability epsilon, select randomly among all actions,
# otherwise, choose greedy action
def e_greedy(q_t_values, epsilon):
  if (np.random.rand() < epsilon):
    return np.random.randint(len(q_t_values))
  else:
    return greedy(q_t_values)