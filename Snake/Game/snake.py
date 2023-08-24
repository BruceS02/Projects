"""
Author: Bruce Smith
Date: August 24, 2023
"""
import pygame
from Game.cellitem import CellItem


class Apple(CellItem):
    """
    Apple object that the snake "eats" to grow
    """
    pass


class Node(CellItem):
    """
    Node object that makes up the Snake linked-list
    """
    def __init__(self, x: int, y: int) -> None:
        """
        Initialize node object
        :param x: width index of node
        :param y: height index of node
        """
        super().__init__(x, y)
        self.next = None


class Snake:
    """
    Snake linked-list class comprised of Nodes
    """
    def __init__(self, head_x: int, head_y: int) -> None:
        """
        Initialize snake with head
        :param head_x: width index of head
        :param head_y: height index of head
        """
        self.head = Node(head_x, head_y)
        self.tail = Node(head_x-1, head_y)
        self.head.next = self.tail
        self.prev_dir = pygame.K_RIGHT
        self.score = 0

    def __len__(self) -> int:
        """
        Get length of snake
        :return: length of snake
        """
        return self.size()

    def get_score(self) -> int:
        """
        Get snake's score, i.e. number of apples eaten
        :return: Score of snake
        """
        return self.score

    def increment_score(self) -> None:
        """
        Increment snake's score by one
        :return: None
        """
        self.score += 1

    def get_coordinates_list(self) -> list[tuple[int,int]]:
        """
        Get list of coordinates of snake nodes
        :return: List of tuples with coordinates for each node
        """
        coordinates = []
        curr_node = self.head
        while curr_node is not None:
            coordinates.append(curr_node.get_coordinates())
            curr_node = curr_node.next
        return coordinates

    def size(self) -> int:
        """
        Calculate number of nodes in snake
        :return: number of nodes in snake
        """
        size = 0
        curr = self.head
        while curr is not None:
            size += 1
            curr = curr.next
        return size

    def move(self, key_cmd: int) -> None:
        """
        Move the snake
        :param key_cmd: pygame constant int to relay key pressed
        :return: None
        """
        prev_coord = self.head.get_coordinates()
        # set new head coordinates
        if key_cmd == pygame.K_UP:
            self._move_up()
        elif key_cmd == pygame.K_DOWN:
            self._move_down()
        elif key_cmd == pygame.K_RIGHT:
            self._move_right()
        elif key_cmd == pygame.K_LEFT:
            self._move_left()
        else:
            self._move_straight()

        # update body coordinates
        curr_node = self.head.next
        while curr_node is not None:
            curr_coord = curr_node.get_coordinates()
            curr_node.set_coordinates(prev_coord)
            prev_coord = curr_coord
            curr_node = curr_node.next

    def extend_and_move(self, key_cmd: int) -> None:
        """
        Extend and move the snake
        :param key_cmd: pygame constant int to relay key pressed
        :return: None
        """
        new_node = Node(self.tail.x, self.tail.y)
        self.move(key_cmd)
        self.tail.next = new_node
        self.tail = new_node

    def _move_up(self) -> None:
        """
        Move snake head up
        :return: None
        """
        if self.prev_dir != pygame.K_DOWN:
            self.head.y -= 1
            self.prev_dir = pygame.K_UP
        else:
            self._move_straight()

    def _move_down(self) -> None:
        """
        Move snake head down
        :return: None
        """
        if self.prev_dir != pygame.K_UP:
            self.head.y += 1
            self.prev_dir = pygame.K_DOWN
        else:
            self._move_straight()

    def _move_right(self) -> None:
        """
        Move snake head right
        :return: None
        """
        if self.prev_dir != pygame.K_LEFT:
            self.head.x += 1
            self.prev_dir = pygame.K_RIGHT
        else:
            self._move_straight()

    def _move_left(self) -> None:
        """
        Move snake head left
        :return: None
        """
        if self.prev_dir != pygame.K_RIGHT:
            self.head.x -= 1
            self.prev_dir = pygame.K_LEFT
        else:
            self._move_straight()

    def _move_straight(self) -> None:
        """
        Move snake in previous direction
        :return: None
        """
        match self.prev_dir:
            case pygame.K_UP:
                self._move_up()
            case pygame.K_DOWN:
                self._move_down()
            case pygame.K_RIGHT:
                self._move_right()
            case pygame.K_LEFT:
                self._move_left()
