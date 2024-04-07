# Flappy Bird Reinforcement Learning Project

## Overview:
This project implements a reinforcement learning agent to play the popular game Flappy Bird. The agent learns to navigate the game environment using the Dueling Double Q Network algorithm with memory recall.

## Methodology:
- **Reinforcement Learning Algorithm:** The agent employs the Dueling Double Q Network algorithm, a variant of the Deep Q-Network (DQN) algorithm.
- **Dueling Double Q Network:** This architecture separates the value and advantage streams of the Q-function to improve learning efficiency and stability.
- **Memory Recall:** The agent uses a memory replay mechanism to store experiences and sample batches for training, facilitating better learning from past experiences.

## Components:
- **Game Environment:** The Flappy Bird game environment is provided by the `ple` (PyGame Learning Environment) library.
- **Agent:** The reinforcement learning agent is implemented using Python and PyTorch. It interacts with the game environment, observes states, takes actions, and learns from experiences.

## Training Process:
1. **Initialization:** The agent initializes its policy and target networks with the Dueling Double Q Network architecture.
2. **Experience Collection:** During gameplay, the agent collects experiences (state, action, reward, next state) and stores them in memory.
3. **Training:** Periodically, the agent samples batches of experiences from memory and updates its Q-network parameters using gradient descent.
4. **Target Network Update:** The target network periodically synchronizes with the policy network to stabilize training.

## Results:
- The agent gradually learns to play Flappy Bird by trial and error, improving its performance over time through reinforcement learning.
- Training progress and performance metrics, such as episode durations or scores, can be visualized and analyzed to track learning effectiveness.
