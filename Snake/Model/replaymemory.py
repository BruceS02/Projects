"""
Author: Bruce Smith
Date: August 24, 2023

Sources used:
 - https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/#
 - https://www.diva-portal.org/smash/get/diva2:1342302/FULLTEXT01.pdf
"""
from collections import namedtuple, deque
import random
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    """
    Replay memory class to help train DQN
    """
    def __init__(self, capacity: int) -> None:
        """
        Initialize replay memory
        :param capacity: number of transitions to hold
        """
        self.memory = deque([], maxlen=capacity)

    def __len__(self) -> int:
        """
        Get number of transitions in memory
        :return: Number of transitions
        """
        return len(self.memory)

    def push(self, state: np.ndarray, action: list[int], next_state: np.ndarray, reward: int, done: bool) -> None:
        """
        Save a transition
        :param state: current state
        :param action: action taken
        :param next_state: next state
        :param reward: reward earned
        :param done: done flag
        :return: None
        """
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size: int) -> list[Transition]:
        """
        Get a random transition sample of size batch_size
        :param batch_size: size of sample
        :return: List of a transitions
        """
        return random.sample(self.memory, batch_size)
