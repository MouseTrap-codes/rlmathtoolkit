# RLMathToolkit

RLMathToolkit is a minimal, math-faithful Python library that implements core reinforcement learning algorithms chapter-by-chapter, following Sutton & Barto's *Reinforcement Learning (2nd Edition)*.


Simulate a basic ε-greedy agent on a 10-armed bandit testbed:

```python
import numpy as np
from rlmathtoolkit.bandits.agents import EpsilonGreedyAgent
from rlmathtoolkit.bandits.envs import NArmedTestbed

# Setup
n = 10
steps = 1000
agent = EpsilonGreedyAgent(n=n, epsilon=0.1)
env = NArmedTestbed(n=n)

rewards = []

# Run simulation
for _ in range(steps):
    action = agent.select_action()
    reward = env.get_reward(action)
    agent.update(action, reward)
    rewards.append(reward)

# Analyze
print(f"Average reward after {steps} steps: {np.mean(rewards):.3f}")
'''


