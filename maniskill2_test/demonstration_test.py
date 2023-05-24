import gym
from tqdm.notebook import tqdm
import numpy as np
import mani_skill2.envs
import h5py
from mani_skill2.utils.io_utils import load_json

env_id = "PickCube-v0"

# Load the trajectory data from the .h5 file
traj_path = f"demos/rigid_body/{env_id}/trajectory.h5"
# You can also replace the above path with the trajectory you just recorded (./tmp/trajectory.h5)
h5_file = h5py.File(traj_path, "r")

# Load associated json
json_path = traj_path.replace(".h5", ".json")
json_data = load_json(json_path)

episodes = json_data["episodes"]  # meta data of each episode
env_info = json_data["env_info"]
env_id = env_info["env_id"]
env_kwargs = env_info["env_kwargs"]

print("env_id:", env_id)
print("env_kwargs:", env_kwargs)
print("#episodes:", len(episodes))


def replay(episode_idx, h5_file, json_data, render_mode="cameras", fps=20):
    episodes = json_data["episodes"]
    ep = episodes[episode_idx]
    # episode_id should be the same as episode_idx, unless specified otherwise
    episode_id = ep["episode_id"]
    traj = h5_file[f"traj_{episode_id}"]

    # Create the environment
    env_kwargs = json_data["env_info"]["env_kwargs"]
    env = gym.make(env_id, **env_kwargs)
    # Reset the environment
    reset_kwargs = ep["reset_kwargs"].copy()
    reset_kwargs["seed"] = ep["episode_seed"]
    env.reset(**reset_kwargs)

    # frames = [env.render(mode=render_mode)]

    for i in range(len(traj["actions"])):
        action = traj["actions"][i]
        obs, reward, done, info = env.step(action)
        print(info)
        # if not IN_COLAB: env.render()
        # frames.append(env.render(mode=render_mode))

    env.close()
    del env
    # return frames


episode_idx = 3
replay(episode_idx, h5_file, json_data)