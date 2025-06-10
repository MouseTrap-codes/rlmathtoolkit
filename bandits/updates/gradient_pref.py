# this is an instance of stochastic gradient ascent!
def gradient_pref_update(pi_t_values, h_t_values, average_reward_t, reward_t, alpha, A_t):
  for a in range(len(h_t_values)):
    if (a == A_t):
        # update preference for selected action
        h_t_values[a] = h_t_values[a] + alpha * (reward_t - average_reward_t) * (1 - pi_t_values[a])
    else:
        # update preferences for all other actions
        h_t_values[a] = h_t_values[a] - alpha * (reward_t - average_reward_t) * pi_t_values[a]