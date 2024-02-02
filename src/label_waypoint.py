import os
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
import h5py

from mani_skill2.utils.io_utils import load_json
import mani_skill2.envs
import torch

from autocot import (
    RecNetConfig,
    KeyNetConfig,
    FutureNetConfig,
    ImplicitSAGPTConfig,
    ExplicitSAGPTConfig,
    ExplicitSAHNGPTConfig,
    ImplicitSAResFCConfig,
    ExplicitSAHNConfig,
    AutoCoT
)

from vec_env import get_mp_envs  # Used for parallel evaluation.

from path import MODEL_PATH, DATA_PATH

@torch.no_grad()
def d_point_line(vl1, vl2, v):
    cos_value = torch.cosine_similarity(v - vl1, vl2 - vl1, dim=-1)
    cos_value = torch.clip(cos_value, -0.999, 0.999)
    sin_value = torch.sqrt(1.0 - cos_value ** 2)
    return torch.norm(v - vl1, dim=-1) * sin_value

@torch.no_grad()
def DP_waypoint(states):
    # states (N, D)
    N, D = states.shape
    # dis mat
    vl1 = states.view(N, 1, 1, D).repeat(1, N, N, 1)
    vl2 = states.view(1, N, 1, D).repeat(N, 1, N, 1)
    v = states.view(1, 1, N, D).repeat(N, N, 1, 1)

    m_dis = d_point_line(vl1, vl2, v)  # (N, N, N)
    mask = torch.arange(N, device=states.device)
    mask1 = mask.view(N, 1, 1).repeat(1, N, N)
    mask2 = mask.view(1, N, 1).repeat(N, 1, N)
    mask3 = mask.view(1, 1, N).repeat(N, N, 1)
    mask = torch.logical_and(torch.less_equal(mask1, mask3), torch.greater_equal(mask2, mask3))
    # mask out of range dis
    m_dis = torch.where(mask, m_dis, float('-inf'))
    m_dis, m_dis_idx = torch.max(m_dis, dim=-1)

    DP_value_list = [m_dis]  # list of (N, N)
    DP_label_1 = torch.arange(N, device=states.device).view(1, -1).repeat(N, 1)
    mask_tri = torch.tril(torch.ones((N, N), device=states.device, dtype=torch.bool))
    DP_label_1 = torch.where(mask_tri, -1, DP_label_1).unsqueeze(-1)
    DP_label_list = [DP_label_1]  # list of (N, N, K)

    for K in range(2, 11, 1):
        DP_value_1 = DP_value_list[0]   # (N, N)
        DP_label_1 = DP_label_list[0]  # (N, N, 1)
        DP_value_K_minus_1 = DP_value_list[-1]  # (N, N)
        DP_label_K_minus_1 = DP_label_list[-1]  # (N, N, K-1)

        DP_value_K = \
            torch.maximum(DP_value_1.view(N, N, 1).repeat(1, 1, N), DP_value_K_minus_1.view(1, N, N).repeat(N, 1, 1))
            # (N, _N_ , N)
        DP_label_K = \
            torch.cat([DP_label_1.view(N, N, 1, 1).repeat(1, 1, N, 1),
                       DP_label_K_minus_1.view(1, N, N, K-1).repeat(N, 1, 1, 1)], dim=-1)
            # (N, _N_ , N, K)
        # (N, N, N)
        mask_now = torch.logical_and(torch.less(mask1, mask2), torch.less(mask2, mask3))
        DP_value_K = torch.where(mask_now, DP_value_K, float('+inf'))
        DP_value_K, idx = torch.min(DP_value_K, dim=1)
        DP_value_list.append(DP_value_K)

        # get DP_label_K
        DP_label_K = torch.gather(DP_label_K, index=idx.view(N, 1, N, 1).repeat(1, 1, 1, K), dim=1).squeeze(1)
        DP_label_list.append(DP_label_K)
        # print(f'K = {K}:', DP_value_K[0, N - 1].item(), DP_label_K[0, N - 1])
        if K == 10:
            return DP_label_K[0, N - 1]
    return


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyper-parameters regarding the demo dataset (used to gather eval_ids)
    parser.add_argument('--task', type=str, default='PegInsertionSide-v0', help="Task (env-id) in ManiSkill2.")
    parser.add_argument('--control_mode', type=str, default='pd_joint_delta_pos',
                        help="Control mode used in envs from ManiSkill2.")
    parser.add_argument('--obs_mode', type=str, default='state',
                        help="State mode used in envs from ManiSkill2.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for data spliting.")
    parser.add_argument("--n_traj", default=100, type=int, help="num of validation trajectory.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Load demos to fetch the env. seeds used in training.
    traj_path = os.path.join(
        DATA_PATH,
        f'{args.task}/trajectory.{args.obs_mode}.{args.control_mode}.h5'
    )
    traj_save_keys_path = os.path.join(DATA_PATH, f'{args.task}')
    dataset = {}
    traj_all = h5py.File(traj_path)
    length = args.n_traj
    if length == -1:
        length = len(traj_all)

    ids = np.arange(length)

    dataset['env_states'] = [np.array(traj_all[f"traj_{i}"]['env_states']) for i in ids]
    dataset['obs'] = [np.array(traj_all[f"traj_{i}"]["obs"]) for i in ids]
    dataset['actions'] = [np.array(traj_all[f"traj_{i}"]["actions"]) for i in ids]

    key_states_gts = list()
    for k in traj_all['traj_0']['infos'].keys():
        dataset[f'infos/{k}'] = [np.array(traj_all[f"traj_{i}"]["infos"][k]) for i in ids]
        if k == 'info':  # For PushChair.
            for kk in traj_all['traj_0']['infos'][k].keys():
                dataset[f'infos/demo_{kk}'] = [np.array(
                    traj_all[f"traj_{i}"]["infos"][k][kk]) for i in ids]
    # If TurnFaucet-v0 (two key states)
    # key state I: is_contacted -> true
    # key state II: end of the trajectory
    if args.task == 'TurnFaucet-v0':
        for idx in range(length):
            key_states_gt = list()
            for step_idx, key in enumerate(dataset['infos/is_contacted'][idx]):
                if key:
                    key_states_gt.append(('is_contacted', step_idx))
                    break
            key_states_gt.append(('end', dataset['env_states'][idx].shape[0] - 1))
            key_states_gts.append(key_states_gt)

    # If PegInsertion (three key states)
    # key state I: is_grasped -> true
    # key state II: pre_inserted -> true
    # key state III: end of the trajectory
    if args.task == 'PegInsertionSide-v0':
        for idx in range(length):
            key_states_gt = list()
            for step_idx, key in enumerate(dataset['infos/is_grasped'][idx]):
                if key:
                    key_states_gt.append(('is_grasped', step_idx))
                    break
            for step_idx, key in enumerate(dataset['infos/pre_inserted'][idx]):
                if key:
                    key_states_gt.append(('pre_inserted', step_idx))
                    break
            key_states_gt.append(('end', dataset['env_states'][idx].shape[0] - 1))
            key_states_gts.append(key_states_gt)

    # If PickCube (two key states)
    # key state I: is_grasped -> true
    # key state II: end of the trajectory
    if args.task == 'PickCube-v0':
        for idx in range(length):
            key_states_gt = list()
            for step_idx, key in enumerate(dataset['infos/is_grasped'][idx]):
                if key:
                    key_states_gt.append(('is_grasped', step_idx))
                    break
            key_states_gt.append(('end', dataset['env_states'][idx].shape[0] - 1))
            key_states_gts.append(key_states_gt)

    # If StackCube (three key states)
    # key state I: is_cubaA_grasped -> true
    # key state II: the last state of is_cubeA_on_cubeB -> true
    #               right before is_cubaA_grasped -> false
    # key state III: end of the trajectory
    if args.task == 'StackCube-v0':
        for idx in range(length):
            key_states_gt = list()
            for step_idx, key in enumerate(dataset['infos/is_cubaA_grasped'][idx]):
                if key:
                    key_states_gt.append(('is_cubaA_grasped', step_idx))
                    break
            for step_idx, k1 in enumerate(dataset['infos/is_cubeA_on_cubeB'][idx]):
                k2 = dataset['infos/is_cubaA_grasped'][idx][step_idx]
                if k1 and not k2:
                    key_states_gt.append(('is_cubeA_on_cubeB', step_idx))
                    break
            key_states_gt.append(('end', dataset['env_states'][idx].shape[0] - 1))
            key_states_gts.append(key_states_gt)

    # If PushChair (four key states):
    # key state I: right before demo_rotate -> true
    # key state II: right before demo_move -> true
    # key state III: when chair_close_to_target & chair_standing -> true
    # key state IV: end of the trajectory
    # In PushChair, demo_* indicate the current state (not the next).
    if args.task == 'PushChair-v1':
        for idx in range(length):
            key_states_gt = list()
            for step_idx, key in enumerate(dataset['infos/demo_rotate'][idx]):
                if key:
                    key_states_gt.append(('demo_rotate', step_idx))
                    break
            for step_idx, key in enumerate(dataset['infos/demo_move'][idx]):
                if key:
                    key_states_gt.append(('demo_move', step_idx))
                    break
            for step_idx, key in enumerate(np.bitwise_and(dataset['infos/chair_close_to_target'][idx],
                                                          dataset['infos/chair_standing'][idx])):
                if key:
                    key_states_gt.append(('chair_close_to_target(chair_standing)', step_idx))
                    break
            key_states_gt.append(('end', dataset['env_states'][idx].shape[0] - 1))
            key_states_gts.append(key_states_gt)



    bias_sum = 0.0

    with open(traj_save_keys_path + '/keys-waypoint.txt', 'w') as fk:
        for i_traj in range(length):
            print(f'#{i_traj}')
            traj_state = dataset['obs'][i_traj]
            key_states_gt = [item[1] for item in key_states_gts[i_traj]]
            print(key_states_gt)
            traj_state = torch.tensor(traj_state)
            traj_state = traj_state.cuda()
            key_line = DP_waypoint(traj_state)
            key_line_list = [key_line[i].item() for i in range(10)]

            key_line_str = ''
            j = 0
            for i in range(10):
                if j < len(key_states_gt) and key_line_list[i] >= key_states_gt[j]:
                    bias_sum += (key_line_list[i] - key_states_gt[j])
                    j += 1
                key_line_str += (str(key_line_list[i]) + ',')
            fk.write(key_line_str + '\n')

    with open('keys-waypoint-log.txt', 'a') as flog:
        flog.write(args.task + ' ' + str(bias_sum / length) + '\n')

