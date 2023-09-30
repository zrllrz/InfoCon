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
    parser.add_argument("--idx", default=0, type=int, help="which idx")
    parser.add_argument("--use_hand_camera", action='store_true', help='use hand camera')

    return parser.parse_args()


if __name__ == "__main__":
    SIDE_WIDTH = 280

    font_path = '/usr/share/fonts/truetype/ubuntu/Ubuntu-C.ttf'

    args = parse_args()

    # Load demos to fetch the env. seeds used in training.
    traj_path = os.path.join(
        DATA_PATH,
        os.path.join(
            args.task,
            'trajectory.state.pd_joint_delta_pos.rgbd.pd_joint_pos.h5'
        )
    )
    traj_path_key = os.path.join(
        DATA_PATH,
        f'{args.task}/trajectory.state.pd_joint_delta_pos.h5'
    )

    traj_save_keys_path = os.path.join(DATA_PATH, f'{args.task}')
    dataset = {}
    traj_all = h5py.File(traj_path)
    length = len(traj_all)
    traj_all_key = h5py.File(traj_path_key)

    ids = np.arange(length)

    # if args.use_hand_camera:
    #     use_camera = 'hand_camera'
    # else:
    #     use_camera = 'base_camera'
    use_hand_camera = args.use_hand_camera

    if use_hand_camera:
        frames = torch.from_numpy(np.array(traj_all[f"traj_{args.idx}"]["obs"]['image']['hand_camera']['rgb']))
    else:
        frames = torch.from_numpy(np.array(traj_all[f"traj_{args.idx}"]["obs"]['image']['base_camera']['rgb']))

    frames[:, :1, :, ] = torch.tensor([[[0, 0, 0]]])
    frames[:, -1:, :, ] = torch.tensor([[[0, 0, 0]]])
    frames[:, :, :1, ] = torch.tensor([[[0, 0, 0]]])
    frames[:, :, -1:, ] = torch.tensor([[[0, 0, 0]]])
    print(frames.shape)

    key_frames = list()
    key_frames_gt = list()

    for k in traj_all_key[f"traj_{args.idx}"]['infos'].keys():
        dataset[f'infos/{k}'] = [np.array(traj_all_key[f"traj_{i}"]["infos"][k]) for i in ids]
        if k == 'info':  # For PushChair.
            for kk in traj_all_key[f"traj_{args.idx}"]['infos'][k].keys():
                dataset[f'infos/demo_{kk}'] = [np.array(
                    traj_all[f"traj_{i}"]["infos"][k][kk]) for i in ids]

    # If TurnFaucet-v0 (two key states)
    # key state I: is_contacted -> true
    # key state II: end of the trajectory
    if args.task == 'TurnFaucet-v0':
        key_states_gt = list()
        for step_idx, key in enumerate(dataset['infos/is_contacted'][args.idx]):
            if key:
                key_states_gt.append(step_idx)
                break
        # key_states_gt.append(dataset['env_states'][args.idx].shape[0] - 1)

    # If PegInsertion (three key states)
    # key state I: is_grasped -> true
    # key state II: pre_inserted -> true
    # key state III: end of the trajectory
    elif args.task == 'PegInsertionSide-v0':
        key_states_gt = list()
        for step_idx, key in enumerate(dataset['infos/is_grasped'][args.idx]):
            if key:
                key_states_gt.append(step_idx)
                break
        for step_idx, key in enumerate(dataset['infos/pre_inserted'][args.idx]):
            if key:
                key_states_gt.append(step_idx)
                break
        # key_states_gt.append(dataset['env_states'][args.idx].shape[0] - 1)

    # If PickCube (two key states)
    # key state I: is_grasped -> true
    # key state II: end of the trajectory
    elif args.task == 'PickCube-v0':
        key_states_gt = list()
        for step_idx, key in enumerate(dataset['infos/is_grasped'][args.idx]):
            if key:
                key_states_gt.append(step_idx)
                break
        # key_states_gt.append(dataset['env_states'][args.idx].shape[0] - 1)

    # If StackCube (three key states)
    # key state I: is_cubaA_grasped -> true
    # key state II: the last state of is_cubeA_on_cubeB -> true
    #               right before is_cubaA_grasped -> false
    # key state III: end of the trajectory
    elif args.task == 'StackCube-v0':
        key_states_gt = list()
        for step_idx, key in enumerate(dataset['infos/is_cubaA_grasped'][args.idx]):
            if key:
                key_states_gt.append(step_idx)
                break
        for step_idx, k1 in enumerate(dataset['infos/is_cubeA_on_cubeB'][args.idx]):
            k2 = dataset['infos/is_cubaA_grasped'][args.idx][step_idx]
            if k1 and not k2:
                key_states_gt.append(step_idx)
                break
        # key_states_gt.append(dataset['env_states'][args.idx].shape[0] - 1)

    # If PushChair (four key states):
    # key state I: right before demo_rotate -> true
    # key state II: right before demo_move -> true
    # key state III: when chair_close_to_target & chair_standing -> true
    # key state IV: end of the trajectory
    # In PushChair, demo_* indicate the current state (not the next).
    elif args.task == 'PushChair-v1':
        key_states_gt = list()
        for step_idx, key in enumerate(dataset['infos/demo_rotate'][args.idx]):
            if key:
                key_states_gt.append(step_idx)
                break
        for step_idx, key in enumerate(dataset['infos/demo_move'][args.idx]):
            if key:
                key_states_gt.append(step_idx)
                break
        for step_idx, key in enumerate(np.bitwise_and(dataset['infos/chair_close_to_target'][args.idx],
                                                      dataset['infos/chair_standing'][args.idx])):
            if key:
                key_states_gt.append(step_idx)
                break
        # key_states_gt.append(dataset['env_states'][args.idx].shape[0] - 1)

    else:
        print('unimplemented task')
        assert False

    for i, ks in enumerate(key_states_gt):
        print(f'#{i}', ks)

    if not os.path.exists("vis_key_states/" + args.task + f"/{args.idx}"):
        os.makedirs("vis_key_states/" + args.task + f"/{args.idx}")
    if not os.path.exists("vis_key_states/" + args.task + "/long"):
        os.makedirs("vis_key_states/" + args.task + "/long")
    if not os.path.exists("vis_key_states/" + args.task + f"/{args.idx}/gt"):
        os.makedirs("vis_key_states/" + args.task + f"/{args.idx}/gt")
    if not os.path.exists("vis_key_states/" + args.task + f"/{args.idx}/pred"):
        os.makedirs("vis_key_states/" + args.task + f"/{args.idx}/pred")

    with open(traj_save_keys_path + '/keys-long3.txt', 'r') as fk:
        key_line = fk.readlines()[args.idx].split(sep=',')
        key_line = [int(item) for item in key_line[:-1]]
        for f_idx, img in enumerate(frames):
            img_torch = img.permute(2, 0, 1).unsqueeze(0)
            if args.task == 'PickCube-v0' or 'StackCube-v0':
                img_torch = img_torch[:, :, 32:-32, 32:-32]
            img_torch = torch.div(img_torch, 255.0)
            img_torch = torch.nn.functional.interpolate(
                input=img_torch,
                size=(512, 512),
                mode='bilinear'
            )
            img_torch = torch.mul(img_torch, 255.0)
            img_torch = img_torch.squeeze(0).permute(1, 2, 0)
            img_torch = img_torch.to(dtype=torch.uint8)
            img = img_torch
            if f_idx in key_line:
                key_frames.append(img)
                # save single
                img_npy = img.cpu().numpy()
                img_int8 = np.uint8(img_npy)
                img_IMAGE = Image.fromarray(img_int8)
                img_IMAGE.save("vis_key_states/" + args.task + f"/{args.idx}/pred/{f_idx}_" + ".jpg")
            if f_idx in key_states_gt:
                img_gt = img.clone()
                key_frames_gt.append(img_gt)
                # save single
                img_npy = img_gt.cpu().numpy()
                img_int8 = np.uint8(img_npy)
                img_IMAGE = Image.fromarray(img_int8)
                img_IMAGE.save("vis_key_states/" + args.task + f"/{args.idx}/gt/{f_idx}_" + ".jpg")


    key_line.sort()
    key_sim = list()
    key_sim_idx = list()
    for i, ks in enumerate(key_states_gt):
        for i, k in enumerate(key_line):
            if k >= ks:
                key_sim.append(k)
                key_sim_idx.append(i)
                break
    lenlen = len(key_line)
    for i in range(lenlen):
        if key_line[i] != -1:
            key_line = key_line[i:]
            for j in range(len(key_sim_idx)):
                key_sim_idx[j] -= i
            break
    print(key_line)
    print(key_states_gt)
    print(key_sim)
    print(key_sim_idx)

    # concat ground truth image
    key_gt_long = list()
    j = 0
    for i in range(len(key_line)):
        if i in key_sim_idx:
            key_gt_long.append(key_frames_gt[j])
            j += 1
        else:
            key_gt_long.append(torch.full(size=(512, 512, 3), fill_value=255, device='cpu'))
        key_gt_long[i][:4, :, ] = torch.tensor([[[0, 0, 0]]])
        key_gt_long[i][-4:, :, ] = torch.tensor([[[0, 0, 0]]])
        key_gt_long[i][:, :4, ] = torch.tensor([[[0, 0, 0]]])
        key_gt_long[i][:, -4:, ] = torch.tensor([[[0, 0, 0]]])
    key_frames_gt_long = torch.cat(key_gt_long, dim=1)
    key_framse_gt_long_txt = torch.full_like(key_frames_gt_long[:128, ...], fill_value=255)
    key_frames_gt_long = torch.cat([key_framse_gt_long_txt, key_frames_gt_long], dim=0)

    # side label
    key_frames_gt_long_side_txt = torch.full(size=(640, SIDE_WIDTH, 3), fill_value=255, device=key_frames_gt_long.device)
    key_frames_gt_long_side_txt[636:, :, ] = torch.tensor([[[0, 0, 0]]], device=key_frames_gt_long.device)
    key_frames_gt_long = torch.cat([key_frames_gt_long_side_txt, key_frames_gt_long], dim=1)


    print(key_frames_gt_long.shape)

    # concat into long image
    # circle nearest key states
    for i in range(len(key_line)):
        if i in key_sim_idx:
            key_frames[i][:12, :, ] = torch.tensor([[[255, 0, 0]]])
            key_frames[i][-12:, :, ] = torch.tensor([[[255, 0, 0]]])
            key_frames[i][:, :12, ] = torch.tensor([[[255, 0, 0]]])
            key_frames[i][:, -12:, ] = torch.tensor([[[255, 0, 0]]])
    key_frames_long = torch.cat(key_frames, dim=1)
    key_framse_long_txt = torch.full_like(key_frames_long[:128, ...], fill_value=255)
    key_frames_long = torch.cat([key_frames_long, key_framse_long_txt], dim=0)
    print(key_frames_long.shape)

    # side label
    key_frames_long_side_txt = torch.full(size=(640, SIDE_WIDTH, 3), fill_value=255, device=key_frames_long.device)
    key_frames_long_side_txt[:4, :, ] = torch.tensor([[[0, 0, 0]]], device=key_frames_long.device)
    key_frames_long = torch.cat([key_frames_long_side_txt, key_frames_long], dim=1)

    key_frames_long = torch.cat([key_frames_gt_long, key_frames_long], dim=0)

    key_frames_long = key_frames_long.cpu().numpy()
    key_frames_long = np.uint8(key_frames_long)
    key_frames_long_img = Image.fromarray(key_frames_long)
    # write frame idx at down part
    draw = ImageDraw.Draw(key_frames_long_img)
    j = 0

    draw.text((SIDE_WIDTH - 90, 512 + 30), 'GT', fill='black', font=ImageFont.truetype(font_path, size=96), align='left')
    draw.text((SIDE_WIDTH - 280, 512 + 120), 'InfoCon', fill='black', font=ImageFont.truetype(font_path, size=96), align='left')
    for i, f_idx in enumerate(key_line):
        text = 'frame ' + str(f_idx)
        draw.text((SIDE_WIDTH + i * 512, 128 + 512 + 512), text, fill='black', font=ImageFont.truetype(font_path, size=96), align='left')
        if i in key_sim_idx:
            text_gt = 'frame ' + str(key_states_gt[j])
            j += 1
            draw.text((SIDE_WIDTH + i * 512, 32), text_gt, fill='black', font=ImageFont.truetype(font_path, size=96), align='left')

    key_frames_long_img.save("vis_key_states/" + args.task + f"/long/{args.idx}_" + ".jpg")

    key_frames_gt_long = key_frames_gt_long.cpu().numpy()
    key_frames_gt_long = np.uint8(key_frames_gt_long)
    key_frames_gt_long_img = Image.fromarray(key_frames_gt_long)
    key_frames_gt_long_img.save("vis_key_states/" + args.task + f"/long/{args.idx}_gt_" + "_gt.jpg")
