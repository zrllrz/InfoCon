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
    ImplicitSAGPTConfig,
    ImplicitSAResFCConfig,
    ExplicitSAHNConfig,
    ENetConfig,
    ActCommitNetConfig,
    AutoCoT
)

from vec_env import get_mp_envs  # Used for parallel evaluation.

from path import MODEL_PATH, DATA_PATH


@torch.no_grad()
def predict(model, action_hist, state_hist, t):
    timesteps = torch.from_numpy(t)[:, None].to(model.device)

    if not action_hist:  # The first step.
        actions = None
    else:
        actions = torch.stack(action_hist, 1).float().to(model.device)
    states = torch.stack(state_hist, 1).float().to(model.device)

    # print('shape of states:', states.shape)
    # print('shape of timesteps', timesteps.shape)

    # use the label_single method in AutoCoT
    indices = model.label_single(states, timesteps, actions)

    return indices


# def update(model, action_hist, state_hist, actions, states, t):
#     # A function used to update the state and action history.
#     actions = torch.from_numpy(actions)
#     if len(state_hist) == model.key_net.block_size // 2:  # The context buffer is full.
#         assert len(action_hist) == model.key_net.block_size // 2 - 1
#         state_hist = state_hist[1:] + [states]
#         action_hist = action_hist[1:] + [actions]
#         t += 1
#     else:
#         state_hist.append(states)
#         action_hist.append(actions)
#     return action_hist, state_hist, t


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

    # Hyper-parameters regarding the module.
    parser.add_argument("--model_name", default='', type=str, help="Model name to be loaded.")
    parser.add_argument("--from_ckpt", default=-1, type=int, help="Ckpt of the module to be loaded.")

    # parser.add_argument("--eval_max_steps", default=200, type=int, help="Max steps allowed in eval.")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    assert args.model_name, 'Should specify --model_name'
    assert args.from_ckpt > 0, 'Should specify --from_ckpt'

    # Load the module.
    path = os.path.join(MODEL_PATH, f'{args.model_name}/epoch{args.from_ckpt}.pth')
    # Load to cpu first to avoid cuda related errors from ManiSkill2.
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    state_dict_from_ckpt, params = ckpt['module'], ckpt['metadata']
    print(state_dict_from_ckpt['key_book.embedding.weight'].shape)

    state_dim = state_dict_from_ckpt['key_net.state_encoder.net.0.weight'].shape[1]
    action_dim = state_dict_from_ckpt['key_net.action_encoder.net.0.weight'].shape[1]
    key_dim = params['dim_key']
    e_dim = params['dim_e']

    # if params['use_key_energy'] is True:
    #     key_dim -= state_dim
    max_timestep = state_dict_from_ckpt['key_net.global_pos_emb'].shape[1]
    print('Loaded ckpt from:', path)
    # Load demos to fetch the env. seeds used in training.
    traj_path = os.path.join(
        DATA_PATH,
        f'{args.task}/trajectory.{args.obs_mode}.{args.control_mode}.h5'
    )
    dataset = {}
    traj_all = h5py.File(traj_path)
    length = args.n_traj
    if length == -1:
        length = len(traj_all)
    np.random.seed(args.seed)
    # If you use the same seed, you can get same trajectory choice
    # Since TurnFaucet uses 10 different faucet models, we shuffle the data
    # such that the resulting sampled data are evenly sampled across faucet models.
    if args.task == 'TurnFaucet-v0':
        ids = []
        for i in range(10):  # Hard-code the 10 data splits for permutation.
            t_ids = np.random.permutation(len(traj_all) // 10)[:length // 10]
            t_ids += i * len(traj_all) // 10
            ids.append(t_ids)
        ids = np.concatenate(ids)
    # Since PushChair uses 5 different faucet models, we shuffle the data
    # such that the resulting sampled data are evenly sampled across chair models.
    elif args.task == 'PushChair-v1':
        ids = []
        for i in range(5):  # Hard-code the 5 data splits for permutation.
            t_ids = np.random.permutation(len(traj_all) // 5)[:length // 5]
            t_ids += i * len(traj_all) // 5
            ids.append(t_ids)
        ids = np.concatenate(ids)
    else:
        ids = np.random.permutation(len(traj_all))[:length]

    dataset['env_states'] = [np.array(traj_all[f"traj_{i}"]['env_states']) for i in ids]
    dataset['obs'] = [np.array(traj_all[f"traj_{i}"]["obs"]) for i in ids]
    dataset['actions'] = [np.array(traj_all[f"traj_{i}"]["actions"]) for i in ids]

    print(dataset['env_states'][0].shape[0])

    # dataset['key_label'] = [[None for j in range(dataset['env_states'][idx].shape[0])] for idx in range(length)]
    # dataset['key_states_gt'] = list()
    key_states_gts = list()

    max_steps = np.max(len(s) for s in dataset['env_states'])

    # print(dataset['env_states'][0].shape, type(dataset['env_states'][0]))
    # print(dataset['obs'][0].shape, type(dataset['obs'][0]))
    # print(dataset['actions'][0].shape, type(dataset['actions'][0]))

    for k in traj_all['traj_0']['infos'].keys():
        dataset[f'infos/{k}'] = [np.array(
            traj_all[f"traj_{i}"]["infos"][k]) for i in ids]
        if k == 'info':  # For PushChair.
            for kk in traj_all['traj_0']['infos'][k].keys():
                dataset[f'infos/demo_{kk}'] = [np.array(
                    traj_all[f"traj_{i}"]["infos"][k][kk]) for i in ids]

    # If TurnFaucet (two key states)
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

    for ksgt in key_states_gts:
        for i, ks in enumerate(ksgt):
            print(f'#{i}', ks[0], ks[1])

    # config our net
    # rec_config = RecNetConfig(
    #     n_embd=params['n_embd'],
    #     n_head=params['n_head'],
    #     attn_pdrop=float(params['dropout']),
    #     resid_pdrop=float(params['dropout']),
    #     embd_pdrop=float(params['dropout']),
    #     block_size=params['context_length'],
    #     n_layer=params['n_key_layer'],
    #     max_timestep=max_timestep,
    # )

    key_config = KeyNetConfig(
        n_embd=params['n_embd'],
        n_head=params['n_head'],
        attn_pdrop=float(params['dropout']),
        resid_pdrop=float(params['dropout']),
        embd_pdrop=float(params['dropout']),
        block_size=params['context_length'],
        n_layer=params['n_key_layer'],
        max_timestep=max_timestep
    )

    if params['sa_type'] == 'resfc':
        sa_config = ImplicitSAResFCConfig(
            n_embd=params['n_embd'],
            block_size=params['context_length'],
            n_state_layer=params['n_state_layer'],
            n_action_layer=params['n_action_layer'],
            max_timestep=max_timestep,
            use_pos_emb=params['use_pos_emb']
        )
    elif params['sa_type'] == 'gpt':
        sa_config = ImplicitSAGPTConfig(
            n_embd=params['n_embd'],
            n_head=params['n_head'],
            attn_pdrop=float(params['dropout']),
            resid_pdrop=float(params['dropout']),
            embd_pdrop=float(params['dropout']),
            block_size=params['context_length'],
            n_layer=params['n_state_layer']+params['n_action_layer'],
            state_layer=params['n_state_layer']-1,
            max_timestep=max_timestep
        )
    elif params['sa_type'] == 'hn':
        sa_config = ExplicitSAHNConfig(
            dim_h=params['n_embd'] * params['n_state_layer'],
            block_size=params['context_length'],
            use_pos_emb=params['use_pos_emb'],
            reward_layer=params['n_state_layer'],
            max_timestep=max_timestep
        )
    else:
        # should not reach here
        print('Unknown sa_type')
        assert False

    # act_config = ActCommitNetConfig(
    #     n_embd=params['n_embd'],
    #     n_head=params['n_head'],
    #     attn_pdrop=float(params['dropout']),
    #     resid_pdrop=float(params['dropout']),
    #     embd_pdrop=float(params['dropout']),
    #     block_size=params['context_length'],
    #     n_layer=params['n_act_layer'],
    #     max_timestep=max_timestep,
    #     commit=False,
    #     use_key_energy=params['use_key_energy']
    # )
    # e_config = ENetConfig(
    #     n_embd=params['n_embd'],
    #     n_head=params['n_head'],
    #     attn_pdrop=float(params['dropout']),
    #     resid_pdrop=float(params['dropout']),
    #     embd_pdrop=float(params['dropout']),
    #     block_size=params['context_length'],
    #     n_layer=params['n_eact_layer'],
    #     max_timestep=max_timestep,
    # )

    print(params['vq_n_e'])
    autocot_model = AutoCoT(
        key_config=key_config,
        sa_config=sa_config,
        vq_n_e=params['vq_n_e'],
        KT=float(params['KT']),
        optimizers_config=None,
        scheduler_config=None,
        state_dim=state_dim,
        action_dim=action_dim,
        key_dim=key_dim,
        e_dim=e_dim
    )

    autocot_model = autocot_model.cuda()
    autocot_model.load_state_dict(state_dict_from_ckpt, strict=True)
    autocot_model.eval()

    for i_traj in range(length):
        traj_state = dataset['obs'][i_traj]
        traj_action = dataset['actions'][i_traj]
        key_states_gt = key_states_gts[i_traj]

        t = np.zeros(shape=[1], dtype=np.int64)
        state_hist, action_hist = [torch.from_numpy(traj_state[:1]).float()], [torch.from_numpy(traj_action[:1]).float()]

        current_label = -1
        i_begin = 0

        for step in range(traj_action.shape[0] - 1):
            # print('step #', step, end=' ')
            indices = predict(
                model=autocot_model,
                action_hist=action_hist,
                state_hist=state_hist,
                t=t,
            )
            # print(indices.item(), traj_label[step])
            indices_item = indices.item()

            # Label output
            if indices_item != current_label:
                if current_label != -1:
                    print(f'key {current_label}\t[{i_begin}, {step - 1}]', end='')
                    for i_gt in range(len(key_states_gt)):
                        if i_begin <= key_states_gt[i_gt][1] <= step - 1:
                            print(f'\tgt key states', key_states_gt[i_gt][1], key_states_gt[i_gt][0], end='')
                    print()
                current_label = indices_item
                i_begin = step

            # update...
            if len(state_hist) == autocot_model.key_net.block_size // 2:
                # print(f'len(state_hist) reach context length{autocot_model.key_net.block_size}')
                assert len(action_hist) == (autocot_model.key_net.block_size // 2)
                state_hist = state_hist[1:] + [torch.from_numpy(traj_state[step + 1:step + 2]).float()]
                action_hist = action_hist[1:] + [torch.from_numpy(traj_action[step: step + 1]).float()]
                t += 1
            else:
                state_hist.append(torch.from_numpy(traj_state[step + 1:step + 2]).float())
                action_hist.append(torch.from_numpy(traj_action[step: step + 1]).float())
        if current_label != -1:
            print(f'key {current_label}\t[{i_begin}, {traj_action.shape[0] - 1}]', end='')
            for i_gt in range(len(key_states_gt)):
                if i_begin <= key_states_gt[i_gt][1] <= (traj_action.shape[0] - 1):
                    print(f'\tgt key states', key_states_gt[i_gt][1], key_states_gt[i_gt][0], end='')
            print()
        input()
