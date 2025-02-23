'''
QUICK NOTES:
1. The epsilon makes the policy explore more, but again if we set epsilon to a high value let's say 0.7 then 
the policy won't be able to exploit much, it will just keep selecting random actions eventhough it has already 
discovered the optimal action and can exploit hence forth.
'''

import numpy as np
import matplotlib.pyplot as plt

# Define the multi-armed bandit problem
np.random.seed(42)  # For reproducibility
num_actions = 5  # Number of arms
true_action_values = np.random.normal(0, 1, num_actions)  # True mean rewards of arms

# Function to simulate the reward of an action
def get_reward(action):
    return np.random.normal(true_action_values[action], 1)  # Reward = N(Q(a), 1)

# Greedy and ε-greedy Bandit Solver
class BanditSolver:
    def __init__(self, num_actions, epsilon=0.1, steps=1000):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.steps = steps
        self.Q_values = np.zeros(num_actions)  # Estimated rewards
        # self.Q_values = np.linspace(0,1,num_actions)
        self.action_counts = np.zeros(num_actions)  # Count of actions taken
        self.total_rewards = np.zeros(steps)  # Store cumulative rewards

    def select_action(self):
        """ Selects an action using ε-greedy policy """
        if np.random.rand() < self.epsilon:  
            return np.random.choice(self.num_actions)  # Explore
        else:
            return np.argmax(self.Q_values)  # Exploit (Greedy choice)

    def update_Q_values(self, action, reward):
        """ Updates Q-values using incremental average """
        self.action_counts[action] += 1
        self.Q_values[action] += (reward - self.Q_values[action]) / self.action_counts[action]

    def run(self):
        """ Runs the bandit simulation """
        for step in range(self.steps):
            action = self.select_action()
            reward = get_reward(action)
            self.update_Q_values(action, reward)
            self.total_rewards[step] = reward
        return np.cumsum(self.total_rewards) / (np.arange(self.steps) + 1)  # Average reward over time

# Run simulations for greedy (ε = 0) and ε-greedy (ε = 0.1)
greedy_solver = BanditSolver(num_actions, epsilon=0.0)
epsilon_greedy_solver = BanditSolver(num_actions, epsilon=0.1)

greedy_rewards = greedy_solver.run()
epsilon_greedy_rewards = epsilon_greedy_solver.run()

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(greedy_rewards, label="Greedy (ε = 0)")
plt.plot(epsilon_greedy_rewards, label="ε-Greedy (ε = 0.1)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()
plt.title("Comparison of Greedy and ε-Greedy Strategies")
plt.show()