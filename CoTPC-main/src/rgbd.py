import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from collections import defaultdict
import h5py

from mani_skill2.utils.io_utils import load_json
import mani_skill2.envs
import torch

from vec_env import get_mp_envs  # Used for parallel evaluation.

from path import MODEL_PATH, DATA_PATH


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyper-parameters regarding the demo dataset (used to gather eval_ids)
    parser.add_argument('--task', type=str, default='PegInsertionSide-v0', help="Task (env-id) in ManiSkill2.")
    parser.add_argument("--idx", default=0, type=int, help="which idx")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Load demos to fetch the env. seeds used in training.
    traj_path = os.path.join(
        DATA_PATH,
        os.path.join(
            args.task,
            'trajectory.state.pd_joint_delta_pos.rgbd.pd_joint_pos.h5'
        )
    )
    traj_save_keys_path = os.path.join(DATA_PATH, f'{args.task}')
    dataset = {}
    traj_all = h5py.File(traj_path)
    length = len(traj_all)

    ids = np.arange(length)

    print(torch.from_numpy(np.array(traj_all[f"traj_{args.idx}"]["obs"]['image']['base_camera']['rgb'])).shape)

    frames = torch.from_numpy(np.array(traj_all[f"traj_{args.idx}"]["obs"]['image']['base_camera']['rgb']))
    key_frames = list()

    with open(traj_save_keys_path + '/keys-0909.txt', 'r') as fk:
        key_line = fk.readlines()[args.idx].split(sep=',')
        key_line = [int(item) for item in key_line[:-1]]
        for f_idx, img in enumerate(frames):
            if f_idx in key_line:
                print(f'append key state at frame {f_idx}')
                key_frames.append(img)

    # save img
    # concat into long image
    key_frames = torch.cat(key_frames, dim=1)
    print(key_frames.shape)
    key_frames = key_frames.cpu().numpy()
    key_frames = np.uint8(key_frames)
    key_frames_img = Image.fromarray(key_frames)

    if not os.path.exists("vis_key_states/" + args.task):
        os.makedirs("vis_key_states/" + args.task)

    key_frames_img.save("vis_key_states/" + args.task + "/test.jpg")
