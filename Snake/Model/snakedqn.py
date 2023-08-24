"""
Author: Bruce Smith
Date: August 24, 2023

Sources used:
 - https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/#
 - https://www.diva-portal.org/smash/get/diva2:1342302/FULLTEXT01.pdf
"""
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from Model.replaymemory import ReplayMemory, Transition
from Model.dqnmodel import QNNTrainer, LinearDQN


class SnakeDQN:

    def __init__(self, lr: float, gamma: float):
        """
        Initialize the Snake DQN
        :param lr: learning rate
        :param gamma: discount rate
        """
        self.memory = ReplayMemory(10000)
        self.model = LinearDQN(11, 256, 3)
        self.trainer = QNNTrainer(self.model, lr, gamma)
        self.epsilon = 1
        self.delta_epsilon = 1e-4
        self.n_game = 0
        self.scores = []
        self.score_means = []

    def get_action(self, state: np.ndarray) -> list[int]:
        """
        Get action to perform (straight, right, left)
        :param state: The state of the GameBoard
        :return: A list specifying which action to perform
        """
        self.epsilon = max(self.epsilon - self.delta_epsilon, 0.0001)
        # [1,0,0] = straight, [0,1,0] = right turn, [0,0,1] = left turn
        final_move = [0, 0, 0]
        if self.epsilon > random.random():
            # get random move (exploration)
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # get DQN predicted move (exploitation)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            print("Prediction= ", prediction)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def train_short_memory(self, state: np.ndarray, action: list[int],
                           next_state: np.ndarray, reward: int, done: bool) -> None:
        """
        Train the DQN with a single game step
        :param state: Current state of board
        :param action: Action performed
        :param next_state: Next state of board
        :param reward: Reward earned
        :param done: True if game ended, False otherwise
        :return: None
        """
        self.trainer.train_step([Transition(state, action, next_state, reward, done)])

    def train_long_memory(self) -> None:
        """
        Train the DQN using a random sample from replay memory
        :return: None
        """
        batch_size = 20
        if len(self.memory) < batch_size:
            sample = self.memory.sample(len(self.memory))
        else:
            sample = self.memory.sample(batch_size)
        self.trainer.train_step(sample)

    def graph_results(self, score: int) -> None:
        """
        Store and graph results of training
        :param score: The current score
        :return: None
        """
        self.scores.append(score)
        self.score_means.append(np.mean(self.scores[-10:]))
        # Graph every 5 games
        if self.n_game % 5 == 0:
            scores = np.array(self.scores)
            n_game = np.arange(0, self.n_game)
            plt.figure(1)
            plt.title('Score vs. number of games')
            plt.xlabel('Game number (n_game)')
            plt.ylabel('Score')
            plt.plot(n_game, scores, label="Score")
            plt.plot(n_game, self.score_means, label="10 game average")
            plt.legend()
            plt.show()

            plt.pause(0.001)
