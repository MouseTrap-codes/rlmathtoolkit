# sample average method
def sample_average_method(q_prev, reward, k):
  return q_prev + (1/k) * (reward - q_prev)