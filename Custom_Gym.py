import gym
import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import random
from gym import Env
from gym.spaces import Discrete, Box
#author Siyuan Chen


class FactsEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # array???
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start state???
        self.state = 38 + random.randint(-3, 3)
        # Set shower length
        # self.shower_length = 60

    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0
        # 2 -1 = 1 temperature
        self.state += action - 1
        # Reduce shower length by 1 second
        # self.shower_length -= 1

        # Calculate reward???
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

            # Check if shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # Apply temperature noise
        # self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        self.state = 38 + random.randint(-3, 3)
        # Reset shower time
        self.shower_length = 60
        return self.state

'''
input_var = numpy.array([3, 3, 2, 2, 7, 8, 4])
input = input_var.tolist()
#print(input_var[0])

numpy.savetxt('input.txt', input_var)

os.system("xsim-runner.exe --model LawMcComasMOPs.xml --input input.txt --output_txt output_Law2.txt")

# check the output array
with open('output_Law2.txt') as my_file:
    # Throughput, Work-In-Process, Parts-Produced, and Lead-Time
    output_array = my_file.readlines()

throughout = float(output_array[0])
print(throughout)
print(output_array)

#calculate the maximum profit
profit = (200 * throughout * 720) - 25000 * (input[0]+input[1]+input[2]+input[3]) - 1000 * (input[4]+input[5]+input[6])
print(profit)
'''