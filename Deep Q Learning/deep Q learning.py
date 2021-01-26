import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")
SIZE = 10
HM_EPISODE = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 3000
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.998
SHOW_EVERY = 3000

start_q_table = None
learning_rate = 0.1
discount = 0.95

player_n = 1
food_n = 2
enemy_n = 3

d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (255, 0, 0)}


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(o, SIZE)

    def __str__(self):
        return f"{self.x},{self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        pass

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


if start_q_table is None:
    q_table = {}
    for i in range(-SIZE+1, SIZE):
