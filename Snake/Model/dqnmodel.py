"""
Author: Bruce Smith
Date: August 24, 2023

Sources used:
 - https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/#
 - https://www.diva-portal.org/smash/get/diva2:1342302/FULLTEXT01.pdf
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Model.replaymemory import Transition


class LinearDQN(nn.Module):
    def __init__(self, input_n: int, hidden_n: int, output_n: int) -> None:
        """
        Initialize DQN model
        :param input_n: size of input
        :param hidden_n: number of hidden nodes
        :param output_n: size of output
        """
        super().__init__()
        self.layer1 = nn.Linear(input_n, hidden_n)
        self.layer2 = nn.Linear(hidden_n, output_n)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """
        Given an input state, determine predicted action outputted by the model
        :param x: input state
        :return: output tensor with predicted action to take
        """
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class QNNTrainer:
    def __init__(self, model: LinearDQN, lr: float, gamma: float) -> None:
        """
        Initialize the DQN trainer
        :param model: the model to use
        :param lr: learning rate
        :param gamma: discount rate
        """
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, transitions: list[Transition]) -> None:
        """
        Train the DQN
        :param transitions: a list of Transitions use to train the model
        :return: None
        """
        batch = Transition(*zip(*transitions))
        state = torch.tensor(batch.state, dtype=torch.float)
        next_state = torch.tensor(batch.next_state, dtype=torch.float)
        action = torch.tensor(batch.action, dtype=torch.long)
        reward = torch.tensor(batch.reward, dtype=torch.float)
        done = torch.tensor(batch.done, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = q_new
        # 2. Q_new = reward + gamma * max(next_predicted Qvalue)

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()  # backward propagation of loss

        self.optimizer.step()
