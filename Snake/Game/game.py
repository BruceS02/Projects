"""
Author: Bruce Smith
Date: August 24, 2023
"""
import numpy as np
import pygame
from Game.gameboard import GameBoard
from Model.snakedqn import SnakeDQN


class Game:
    """ Game class to run the game """
    def __init__(self, width: int = 800, height: int = 700, dqn: SnakeDQN = None) -> None:
        """
        Initialize the game
        :param width: width of board
        :param height: height of board
        """
        self.width = width
        self.height = height
        self.board = GameBoard(width, height)
        self.key_cmd = -1
        self.game_over = False

        self.dqn = dqn
        self.qnn_play = False if dqn is None else True

    def score(self) -> int:
        """
        Get score of the game
        :return: score
        """
        return self.board.snake.get_score()

    def reset(self) -> None:
        """
        Reset/restart the game
        :return: None
        """
        self.board = GameBoard(self.width, self.height)
        self.key_cmd = -1
        self.game_over = False

    def dqn_pre_move(self) -> tuple[int, np.ndarray, list[int], float]:
        """
        Get pre move data needed for DQN training
        :return: Current score, state, action, and
            euclidean distance between snake head and apple
        """
        # pre-move characteristics
        score = self.score()
        state = self.board.get_state()
        action = self.dqn.get_action(state)
        head_to_apple = self.board.distance_head_to_apple()
        self.key_cmd = self.board.parse_dqn_action(action)
        return score, state, action, head_to_apple

    def dqn_post_move(self, score: int, state: np.ndarray, action: list[int], head_to_apple: int) -> None:
        """
        Get post move data and train DQN model
        :param score: Score before move
        :param state: State before move
        :param action: Action taken
        :param head_to_apple: Euclidean distance from head to apple before move
        :return: None
        """
        # get next score, next state, and reward
        next_state = self.board.get_state()
        next_score = self.score()
        next_head_to_apple = self.board.distance_head_to_apple()
        done = False
        if self.game_over:
            reward = -10
            done = True
        elif next_score - score > 0:
            reward = 10
        elif next_head_to_apple < head_to_apple:
            reward = 1
        else:
            reward = 0

        self.dqn.train_short_memory(state, action, next_state, reward, done)
        self.dqn.memory.push(state, action, next_state, reward, done)

    def play(self) -> None:
        """
        Run the game
        :return: None
        """
        pygame.init()
        self.board = GameBoard()
        pygame.display.set_caption("Snake: version by Bruce Smith")

        # define move event timer
        MOVEEVENT = pygame.USEREVENT+1
        pygame.time.set_timer(MOVEEVENT, 100)

        running = True
        while running:
            # check events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and not self.qnn_play:
                    self.key_cmd = event.key
                if self.key_cmd == pygame.K_q:
                    running = False
                if self.key_cmd == pygame.K_r:
                    self.reset()
                if not self.game_over:
                    if event.type == MOVEEVENT:
                        # DQN pre-move
                        if self.qnn_play:
                            score, state, action, head_to_apple = self.dqn_pre_move()
                        # normal game procedure
                        if self.board.apple_collision():
                            self.game_over = not self.board.extend_and_move_snake(self.key_cmd)
                        else:
                            self.game_over = not self.board.move_snake(self.key_cmd)
                        # DQN post-move
                        if self.qnn_play:
                            self.dqn_post_move(score, state, action, head_to_apple)

                        self.key_cmd = -1

            self.board.draw(self.game_over)
            # If game_over, graph game results and train qnn with memory sample
            if self.qnn_play and self.game_over:
                self.dqn.n_game += 1
                self.dqn.graph_results(self.score())
                self.reset()
                self.dqn.train_long_memory()

            pygame.display.update()


if __name__ == "__main__":
    dqn_play = True
    if dqn_play:
        # Train Deep-Q neural network to play the game
        dqn_snake = SnakeDQN(lr=0.001, gamma=0.97)
        game = Game(dqn=dqn_snake)
        game.play()
    else:
        # Play the game yourself
        game = Game()
        game.play()
