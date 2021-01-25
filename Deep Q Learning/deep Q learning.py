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
