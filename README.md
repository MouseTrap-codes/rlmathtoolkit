# RLMathToolkit

RLMathToolkit is a minimal, math-faithful Python library that implements core reinforcement learning algorithms chapter-by-chapter, following Sutton & Barto's *Reinforcement Learning (2nd Edition)*.

## Running Experiments

To reproduce the figures from Sutton & Barto (Chapter 2), clone the repo and run any of the experiment scripts:

```bash
git clone https://github.com/MouseTrap-codes/rlmathtoolkit.git
cd rlmathtoolkit
python -m experiments.experiments_02_bandits/fig_two_one.py
```
## Usage Example

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
```



