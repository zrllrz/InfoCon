import os
import numpy as np
import argparse
import h5py

from path import DATA_PATH


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyper-parameters regarding the demo dataset (used to gather eval_ids)
    parser.add_argument('--task', type=str, default='PegInsertionSide-v0', help="Task (env-id) in ManiSkill2.")
    parser.add_argument('--control_mode', type=str, default='pd_joint_delta_pos',
                        help="Control mode used in envs from ManiSkill2.")
    parser.add_argument('--obs_mode', type=str, default='state',
                        help="State mode used in envs from ManiSkill2.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for data spliting.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    traj_path = os.path.join(
        DATA_PATH,
        f'{args.task}/trajectory.{args.obs_mode}.{args.control_mode}.h5'
    )
    traj_save_keys_path = os.path.join(DATA_PATH, f'{args.task}')
    dataset = {}
    traj_all = h5py.File(traj_path)
    length = len(traj_all)
    ids = np.arange(length)

    dataset['env_states'] = [np.array(traj_all[f"traj_{i}"]['env_states']) for i in ids]
    dataset['obs'] = [np.array(traj_all[f"traj_{i}"]["obs"]) for i in ids]
    dataset['actions'] = [np.array(traj_all[f"traj_{i}"]["actions"]) for i in ids]

    key_states_gts = list()

    max_steps = np.max(len(s) for s in dataset['env_states'])

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

    with open(traj_save_keys_path + '/keys-clip.txt', 'r') as fk:
        for i_traj in range(length):
            # gt key states
            key_states_gt = key_states_gts[i_traj]
            # read from line
            line = fk.readline()
            line = line.split(sep=',')[:-1]
            line_int = list()
            for i in range(len(line)):
                if int(line[i]) != -1:
                    line_int.append(int(line[i]))
            for ks in key_states_gt:
                line_int_sub = [(item - ks[1]) for item in line_int]
                min_next = max(line_int_sub)
                for item in line_int_sub:
                    if item >= 0 and item < min_next:
                        min_next = item
                bias_sum += min_next

    print('average bias', bias_sum / length)
