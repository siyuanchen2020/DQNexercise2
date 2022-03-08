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


# callback class for the cartpole
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)
        return True

#models_dir = "/models/dqn"
log_dir = "/tmp/gym/"
#os.makedirs(log_dir, exist_ok=True)

'''
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
'''

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#make the gym environment and monitor it
env = gym.make("CartPole-v0")
env = Monitor(env, log_dir)

#callback using the class before
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# generate the model by DQN
model = DQN("MlpPolicy", env, verbose=1)
# use the model learning and set the total timesteps and callback
model.learn(total_timesteps=100000, log_interval=4, callback=callback)
# save the model
model.save("dqn_cartpole")
#check the type of the model
print(type(model))
#check the mean-reward and standard deviation of the reward using the evaluation policy
mean_reward_after, std_reward_after = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward_after:.2f} +/- std_reward:{std_reward_after:.2f}")

#del model # remove to demonstrate saving and loading

#model = DQN.load("dqn_cartpole")

obs = env.reset()

#check how the model runs
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

# check the action space and observation space in the environment
print(env.action_space)
print(env.observation_space)

print(env.action_space.sample())
print(env.observation_space.shape)
print(env.observation_space.sample())

print(log_dir)

#plot the result with X axis timesteps and y axis reward
results_plotter.plot_results([log_dir], 1000000, results_plotter.X_TIMESTEPS, "DQN CartPole-v0")
#plt.show()

# plot a smoother figure
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

plot_results(log_dir)

#close the environment
env.close()
