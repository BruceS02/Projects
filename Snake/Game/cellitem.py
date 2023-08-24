"""
Author: Bruce Smith
Date: August 24, 2023
"""


class CellItem:
    """
    CellItem object that defines an object in a cell
    """
    def __init__(self, x: int, y: int) -> None:
        """
        Initialize CellItem object
        :param x: width index of item
        :param y: height index of item
        """
        self.x = x
        self.y = y

    def get_coordinates(self) -> tuple[int, int]:
        """
        Get item coordinates
        :return: tuple of coordinates (x, y)
        """
        return self.x, self.y

    def set_coordinates(self, coords: tuple[int, int]) -> None:
        """
        Set item coordinates
        :param coords: tuple of coordinates (x, y)
        :return: None
        """
        self.x = coords[0]
        self.y = coords[1]
