# RLMathToolkit
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


RLMathToolkit is a minimal, math-faithful Python library that implements core reinforcement learning algorithms chapter-by-chapter, following [Sutton & Barto's *Reinforcement Learning (2nd Edition)*](http://incompleteideas.net/book/the-book-2nd.html).

## Installation

You can get started by cloning the repository and installing the package locally:

```bash
# Clone the repo
git clone https://github.com/MouseTrap-codes/rlmathtoolkit.git
cd rlmathtoolkit

# (Optional but recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install .
```

### 🔚 Exiting the virtual environment

When you're done using the toolkit, you can deactivate the virtual environment:

```bash
deactivate
```


## Running Experiments

To reproduce a specific figure from Sutton & Barto, run the corresponding script in the rlmathtoolkit/bandits/experiments/ folder.

For example, to generate Figure 2.1:
```bash
python experiments/experiments_02_bandits/fig_two_one.py
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
## 🛣️ Roadmap

RLMathToolkit currently focuses on core ideas from **Chapter 2 (Multi-Armed Bandits)** of Sutton & Barto. Implementations of future chapters—including value prediction, dynamic programming, Monte Carlo methods, and temporal-difference learning—are planned and will be added incrementally.

Planned additions:

- Chapter 3: Finite Markov Decision Processes (MDPs)
- Chapter 4: Dynamic Programming
- Chapter 5: Monte Carlo Methods
- Chapter 6: Temporal-Difference Learning
- ... and more

Stay tuned for updates!

## 📄 License

MIT



