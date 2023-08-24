"""
Author: Bruce Smith
Date: August 24, 2023
"""
import pygame
from Game.snake import Snake, Apple
from Game.cellitem import CellItem
import random
import numpy as np


class GameBoard:
    """
    GameBoard class
    """
    def __init__(self, width: int = 800, height: int = 700, cell_size: int = 50) -> None:
        """
        Initialize the GameBoard
        :param height: height of window
        :param width: width of window
        """
        self.screen = pygame.display.set_mode((width, height))
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.snake = Snake(int(width/(cell_size*2)) - 2, int(height/(cell_size*2)))
        self.apple = Apple(self.snake.head.x + 3, self.snake.head.y)
        self.colors = {"white": (255, 255, 255),
                       "black": (0, 0, 0),
                       "light_green": (48, 222, 112),
                       "dark_green": (16, 120, 47),
                       "crimson": (220, 20, 60)}

    def draw(self, game_over: bool) -> None:
        """
        Draw all board objects, draw game over screen if game_over
        :param game_over: True if game_over, False otherwise
        :return: None
        """
        self._draw_grid()
        self._draw_snake()
        self._draw_apple()
        self._draw_score()
        if game_over:
            self._draw_game_over()

    def move_snake(self, key_cmd: int) -> bool:
        """
        Update the snakes positions
        :param key_cmd: pygame constant int to relay key pressed
        :return: True if snake position updated, False otherwise
        """
        if not self.is_collision(self.snake.head):
            self.snake.move(key_cmd)
            self.draw(False)
            return not self.is_collision(self.snake.head)
        else:
            return False

    def extend_and_move_snake(self, key_cmd: int) -> bool:
        """
        Extend and update the snakes position
        :param key_cmd: pygame constant int to relay key pressed
        :return: True if snake position, False otherwise
        """
        if not self.is_collision(self.snake.head):
            self.snake.extend_and_move(key_cmd)
            self.apple.set_coordinates(self._get_new_apple_coords())
            self.snake.increment_score()
            self.draw(False)
            return not self.is_collision(self.snake.head)
        else:
            return False

    def apple_collision(self) -> bool:
        """
        Check if snake head collided with apple
        :return: True if collisions occurred, False otherwise
        """
        return self.snake.head.get_coordinates() == self.apple.get_coordinates()

    def is_collision(self, item: CellItem) -> bool:
        """
        Check if there is a collision
        :return: True if collision, False otherwise
        """
        snake_coords = self.snake.get_coordinates_list()
        border_collision = not (0 < item.x < (self.width-self.cell_size)/self.cell_size and
                                0 < item.y < (self.height-self.cell_size)/self.cell_size)
        body_collision = item.get_coordinates() in snake_coords[1:]
        return border_collision or body_collision

    def _draw_game_over(self) -> None:
        """
        Draw the game over text
        :return: None
        """
        font = pygame.font.Font('freesansbold.ttf', 64)
        text = font.render("GAME OVER", True, self.colors["white"], self.colors["black"])
        text_rect = text.get_rect()
        text_rect.center = (self.width / 2, self.height / 2 - self.cell_size*2)
        self.screen.blit(text, text_rect)

        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render("Score: {}".format(self.snake.get_score()), True, self.colors["white"], self.colors["black"])
        text_rect = text.get_rect()
        text_rect.center = (self.width / 2, self.height / 2)
        self.screen.blit(text, text_rect)

        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render("R to restart           Q to quit".format(self.snake.get_score()), True, self.colors["white"], self.colors["black"])
        text_rect = text.get_rect()
        text_rect.center = (self.width / 2, self.height / 2 + self.cell_size)
        self.screen.blit(text, text_rect)

    def _get_new_apple_coords(self) -> tuple[int, int]:
        """
        Get coordinates for new apple location
        :return: Tuple of coordinates
        """
        new_coords = (random.randint(1, int((self.width-self.cell_size)/self.cell_size - 1)),
                      random.randint(1, int((self.height-self.cell_size)/self.cell_size - 1)))
        while new_coords in self.snake.get_coordinates_list():
            new_coords = (random.randint(1, int((self.width-self.cell_size)/self.cell_size - 1)),
                          random.randint(1, int((self.height-self.cell_size)/self.cell_size - 1)))
        return new_coords

    def _draw_grid(self) -> None:
        """
        Draw the grid
        :return: None
        """
        self.screen.fill(self.colors["black"])
        for x in range(self.cell_size, self.width-self.cell_size, self.cell_size):
            for y in range(self.cell_size, self.height-self.cell_size, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.colors["light_green"], rect, 1)

    def _draw_snake(self) -> None:
        """
        Draw the snake
        :return: None
        """
        coordinates = self.snake.get_coordinates_list()
        for x, y in coordinates:
            rect = pygame.Rect(x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size)
            if (x, y) == coordinates[0]:
                pygame.draw.rect(self.screen, self.colors["light_green"], rect)
            else:
                pygame.draw.rect(self.screen, self.colors["dark_green"], rect)

    def _draw_apple(self) -> None:
        """
        Draw apple
        :return: None
        """
        coords = self.apple.get_coordinates()
        pygame.draw.circle(self.screen, self.colors["crimson"],
                           (coords[0]*self.cell_size + 0.5*self.cell_size, coords[1]*self.cell_size + 0.5*self.cell_size),
                           self.cell_size/2)

    def _draw_score(self) -> None:
        """
        Draw score
        :return: None
        """
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render("Score: {}".format(self.snake.get_score()), True, self.colors["white"])
        text_rect = text.get_rect()
        text_rect.center = (self.width/2, self.cell_size/2)
        self.screen.blit(text, text_rect)

# **********************************
# Functions for Deep Q-learning network
# **********************************

    def get_state(self) -> np.ndarray:
        """
        Get current state of the board
        :return: A numpy array of ints describing the board state
        """
        head = self.snake.head

        pt_l = CellItem(head.x - 1, head.y)
        pt_r = CellItem(head.x + 1, head.y)
        pt_u = CellItem(head.x, head.y - 1)
        pt_d = CellItem(head.x, head.y + 1)

        dir_l = self.snake.prev_dir == pygame.K_LEFT
        dir_r = self.snake.prev_dir == pygame.K_RIGHT
        dir_u = self.snake.prev_dir == pygame.K_UP
        dir_d = self.snake.prev_dir == pygame.K_DOWN

        state = [
            # Danger Straight
            (dir_u and self.is_collision(pt_u)) or
            (dir_d and self.is_collision(pt_d)) or
            (dir_l and self.is_collision(pt_l)) or
            (dir_r and self.is_collision(pt_r)),

            # Danger right
            (dir_u and self.is_collision(pt_r)) or
            (dir_d and self.is_collision(pt_l)) or
            (dir_l and self.is_collision(pt_u)) or
            (dir_r and self.is_collision(pt_d)),

            # Danger Left
            (dir_u and self.is_collision(pt_l)) or
            (dir_d and self.is_collision(pt_r)) or
            (dir_l and self.is_collision(pt_d)) or
            (dir_r and self.is_collision(pt_u)),

            # Move Direction
            dir_u,
            dir_d,
            dir_l,
            dir_r,

            # Food Location
            self.apple.x < head.x,  # food is in left
            self.apple.x > head.x,  # food is in right
            self.apple.y < head.y,  # food is up
            self.apple.y > head.y  # food is down
        ]
        return np.array(state, dtype=int)

    def parse_dqn_action(self, action: list[int]) -> int:
        """
        Get a key_cmd from a given action list
        :param action: List of ints specifying the action
        :return: The key_cmd
        """
        key_cmd = -1
        if np.argmax(action) == 0:
            key_cmd = -1
        elif np.argmax(action) == 1:
            match self.snake.prev_dir:
                case pygame.K_RIGHT:
                    key_cmd = pygame.K_DOWN
                case pygame.K_LEFT:
                    key_cmd = pygame.K_UP
                case pygame.K_UP:
                    key_cmd = pygame.K_RIGHT
                case pygame.K_DOWN:
                    key_cmd = pygame.K_LEFT
        elif np.argmax(action) == 2:
            match self.snake.prev_dir:
                case pygame.K_RIGHT:
                    key_cmd = pygame.K_UP
                case pygame.K_LEFT:
                    key_cmd = pygame.K_DOWN
                case pygame.K_UP:
                    key_cmd = pygame.K_LEFT
                case pygame.K_DOWN:
                    key_cmd = pygame.K_RIGHT
        return key_cmd

    def distance_head_to_apple(self) -> float:
        """
        Get the euclidean distance between the snake head and apple
        :return: euclidean distance
        """
        return np.sqrt((self.snake.head.x - self.apple.x)**2 +
                       (self.snake.head.y - self.apple.y)**2)
