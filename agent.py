import torch
import numpy as np
import random
from itertools import count
import pygame as pg
import matplotlib.pyplot as plt
import torch.optim as optim
import memory_recall
import model

class Agent():
    def __init__(self, BATCH_SIZE, MEMORY_SIZE, GAMMA, input_dim, output_dim, action_dim, action_dict, EPS_START, EPS_END, EPS_DECAY_VALUE, lr, TAU, network_type='DDQN') -> None:
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.MEMORY_SIZE=MEMORY_SIZE
        self.action_dim = action_dim
        self.action_dict = action_dict
        self.EPS_START=EPS_START
        self.EPS_END=EPS_END
        self.EPS_DECAY_VALUE=EPS_DECAY_VALUE
        self.eps = EPS_START
        self.TAU = TAU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.episode_durations = []
        self.score=[]
        self.cache_recall = memory_recall.MemoryRecall(memory_size=MEMORY_SIZE)
        self.network_type = network_type
        self.highest_score = 0

        self.policy_net = model.DQN(input_dim=input_dim, output_dim=output_dim, network_type=network_type).to(self.device)
        self.target_net = model.DQN(input_dim=input_dim, output_dim=output_dim, network_type=network_type).to(self.device)
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.steps_done = 0

        
    @torch.no_grad()
    def take_action(self, state):
        self.eps = self.eps*self.EPS_DECAY_VALUE
        self.eps = max(self.eps, self.EPS_END)
        if self.eps < np.random.rand():
            state = state[None, :]
            action_idx = torch.argmax(self.policy_net(state), dim=1).item()
        else:
            action_idx = random.randint(0, self.action_dim-1)
        self.steps_done += 1
        return action_idx
    
    def plot_stats(self):
     plt.figure(figsize=(10, 5))
     plt.plot(range(1, len(self.score) + 1), self.score, marker='o', linestyle='-')
     plt.title('Score vs Episode')
     plt.xlabel('Episode')
     plt.ylabel('Score')
     plt.grid(True)
     plt.savefig('score_vs_episode_plot.png')
     plt.figure(figsize=(10, 5))
     plt.plot(range(1, len(self.episode_durations) + 1), self.episode_durations, marker='o', linestyle='-')
     plt.title('Episode Duration vs Episode')
     plt.xlabel('Episode')
     plt.ylabel('Duration')
     plt.grid(True)
     plt.savefig('duration_vs_episode_plot.png')
     plt.show()
     


    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self):
        if len(self.cache_recall) < self.BATCH_SIZE:
            return
        batch = self.cache_recall.recall(self.BATCH_SIZE)
        batch = [*zip(*batch)]
        state = torch.stack(batch[0])
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[1])), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch[1] if s is not None])
        action = torch.stack(batch[2])
        reward = torch.cat(batch[3])
        next_state_action_values = torch.zeros(self.BATCH_SIZE, dtype=torch.float32, device=self.device)
        state_action_values = self.policy_net(state).gather(1, action)
        with torch.no_grad():
            next_state_action_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_action_values * self.GAMMA) + reward
        loss_fn = torch.nn.SmoothL1Loss()
        loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes, env):
        self.steps_done = 0
        for episode in range(episodes):
            env.reset_game()
            state = env.getGameState()
            state = torch.tensor(list(state.values()), dtype=torch.float32, device=self.device)
            for c in count():
                action = self.take_action(state)
                reward = env.act(self.action_dict[action])
                reward = torch.tensor([reward], device=self.device)
                action = torch.tensor([action], device=self.device)
                next_state = env.getGameState()
                next_state = torch.tensor(list(next_state.values()), dtype=torch.float32, device=self.device)
                done = env.game_over()
                if done:
                    next_state = None
                self.cache_recall.cache((state, next_state, action, reward, done))
                state = next_state
                self.optimize_model()
                self.update_target_network()
                pg.display.update()
                if done:
                    curr_score = env.score()
                    self.episode_durations.append(c+1)
                    self.score.append(curr_score)
                    print("EPS: ",self.eps)
                    print("Durations: ",c+1)
                    print("Score: ",curr_score)
                    if curr_score > self.highest_score:
                        self.highest_score = curr_score
                    torch.save(self.target_net.state_dict(), self.network_type+'_target_net.pt')
                    torch.save(self.policy_net.state_dict(), self.network_type+'_policy_net.pt')
                    break