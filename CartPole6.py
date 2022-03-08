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

# for soumi:
# this python file is just a test to check whether can I use the tensorboard command to generate different plots which I need.
# tensorboard command in terminal which I use:
# tensorboard --logdir = logs

# define where store the model and logs
models_dir = "models"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#make the environment
env = gym.make("CartPole-v0")
env.reset()

#use the DQN algorithm generate the model
model = DQN("MlpPolicy", env, verbose=1)
#set the timesteps
TIMESTEPS = 10000

for i in range(1,30):
    model.learn(total_timesteps= TIMESTEPS, reset_num_timesteps=False, tb_log_name='DQN')
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()