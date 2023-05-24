import gym
import gym.spaces as spaces
from tqdm.notebook import tqdm
import numpy as np
import mani_skill2.envs
import matplotlib.pyplot as plt
import torch.nn as nn
import torch as th

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from mani_skill2.utils.wrappers import RecordEpisode
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3 import PPO

env_id = "LiftCube-v0"
obs_mode = "rgbd"
control_mode = "pd_ee_delta_pose"
reward_mode = "dense"
# create our environment with our configs and then reset to a clean state
env = gym.make(env_id, obs_mode=obs_mode, reward_mode=reward_mode, control_mode=control_mode)
obs = env.reset()

# take a look at the current state
img = env.render(mode="cameras")
plt.imshow(img)
env.close()

# the observations
print("The raw observation", obs.keys())
print("The data in the observation:")
print("image", obs["image"].keys())
print("agent", obs["agent"].keys())
print("extra", obs["extra"].keys())


