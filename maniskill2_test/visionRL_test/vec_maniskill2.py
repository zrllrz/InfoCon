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

from mani_skill2.vector import VecEnv, make as make_vec_env
num_envs = 2 # recommended value for Google Colab. If you have more cores and a more powerful GPU you can increase this
env: VecEnv = make_vec_env(
    env_id,
    num_envs,
    obs_mode=obs_mode,
    reward_mode=reward_mode,
    control_mode=control_mode,
)
env.seed(0)
obs = env.reset()
print("Base Camera RGB:", obs['image']['base_camera']['rgb'].shape)
print("Base Camera RGB device:", obs['image']['base_camera']['rgb'].device)
env.close()
