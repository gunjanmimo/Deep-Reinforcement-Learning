import gym
from gym import wrappers

env = gym.make("CartPole-v0")
env = wrappers.Monitor(env, 'my_unique_folder')

observation = env.reset()

while not done: 
    action = choose_action()
    observation, reward, done, info = env.step(action)
    if done:
        break