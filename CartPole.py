import gym
import random
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import warnings

warnings.filterwarnings("ignore")
LR = 1e-3
env = gym.make("CartPole-v0")
# init env
env.reset()
goal_step = 500
score_requirement = 50
initial_games = 10000


def some_random_games_first():
    # episode of each game
    for episode in range(5):
        env.reset()
        # each frame of each episode
        for t in range(200):
            # render game 
            env.render()
            # sample action space
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                break


some_random_games_first()


def initial_population():
    # [obs, moves]
    training_data = []
    # all scores
    scores = []
    # just the score that met out threshold
    accepted_score = []
    # iterate through how many games we want
    for _ in range(initial_games):
        score = 0
        # moves specifically from this env
        game_memory = []
        # preserve the observation
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_step):
            # choose random action (0 or 1)
            action = random.randrange(0, 2)
            # do the step
            observation, reward, done, info = env.step(action)
            if len(observation) > 0:
                game_memory.append([prev_observation, observation])
            prev_observation += reward
            score += reward
            if done: break
        if score >= score_requirement:
            accepted_score.append(score)
            for data in game_memory:
                # convert data to one-hot
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                # saving training data
                training_data.append([data[0], output])
        # reset env
        env.reset()
        scores.append(score)
