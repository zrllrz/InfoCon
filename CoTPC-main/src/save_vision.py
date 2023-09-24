import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
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
    parser.add_argument("--use_hand_camera", action='store_true', help='use hand camera')

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

    dataset = {}
    print('reading traj')
    traj_all = h5py.File(traj_path)
    print('done')
    length = len(traj_all)

    ids = np.arange(length)
    use_hand_camera = args.use_hand_camera

    if not os.path.exists(os.path.join(DATA_PATH, os.path.join(args.task, 'frames'))):
        os.makedirs(os.path.join(DATA_PATH, os.path.join(args.task, 'frames')))


    for i_traj in range(length):
        frames = np.array(traj_all[f"traj_{i_traj}"]["obs"]['image']['base_camera']['rgb'])
        frames = torch.from_numpy(frames)
        frames = frames.cuda()
        print(f'save traj. {i_traj}\t, framse.shape:', frames.shape)
        frames_save_path = os.path.join(DATA_PATH, os.path.join(args.task, os.path.join('frames', str(i_traj))))
        if not os.path.exists(frames_save_path):
            os.makedirs(frames_save_path)
        for f_idx, img in enumerate(frames):
            img_torch = img.permute(2, 0, 1).unsqueeze(0)
            img_torch = torch.div(img_torch, 255.0)
            img_torch = torch.nn.functional.interpolate(
                input=img_torch,
                size=(512, 512),
                mode='bilinear'
            )
            img_torch = torch.mul(img_torch, 255.0)
            img_torch = img_torch.squeeze(0).permute(1, 2, 0)
            img_torch = img_torch.to(dtype=torch.uint8)
            img_npy = img_torch.cpu().numpy()
            img_int8 = np.uint8(img_npy)
            img_IMAGE = Image.fromarray(img_int8)
            img_IMAGE.save(frames_save_path + f'/{f_idx}.jpg')