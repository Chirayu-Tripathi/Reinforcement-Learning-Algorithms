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
        self.action_counts = np.zeros(num_actions)  # Count of actions taken
        self.total_rewards = np.zeros(steps)  # Store cumulative rewards

    def select_action(self):
        """ Selects an action using greedy policy """
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

# Run simulations for greedy (ε = 0)
greedy_solver = BanditSolver(num_actions, epsilon=0.0)
greedy_rewards = greedy_solver.run()
