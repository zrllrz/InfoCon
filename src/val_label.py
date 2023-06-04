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
    KeyNetConfig,
    ActNetConfig,
    AutoCoT,
)

from vec_env import get_mp_envs  # Used for parallel evaluation.

from path import MODEL_PATH, DATA_PATH

@torch.no_grad()
def predict(model, action_hist, state_hist, t):

    timesteps = torch.from_numpy(t)[:, None].cuda()
    if not action_hist:  # The first step.
        actions = None
    else:
        actions = torch.stack(action_hist, 1).float().to(model.device)
    states = torch.stack(state_hist, 1).float().to(model.device)

    # print('actions device:', actions.device)
    # print('states device:', states.device)
    # print('model device:', model.device)

    # must have cot in act net, but config param should change
    # T is the max sequence size; S is the current number of steps.
    B, T = states.shape[0], model.key_net.block_size + model.act_net.len_key_states
    n_head, S = model.act_net.config.n_head, states.shape[1] - 1  # Exclude the init state.

    # Masks for the all-to-all key state query tokens in attention layers.
    # The built-in masks for causal (auto-regressive) tokens are in `module.py`.
    key_state_mask = torch.zeros([B, n_head, T, T], dtype=torch.bool)
    m1 = torch.arange(0, T).repeat(B, 1)
    m2 = torch.ones([B, 1]) * (S * 2 + model.act_net.len_key_states)
    m3 = m1 > m2  # Tokens in the future are masked out.
    m3 = m3[:, None, None, :].repeat(1, n_head, model.act_net.len_key_states, 1)
    key_state_mask[:, :, :model.len_key_states, :] = m3
    key_state_mask = key_state_mask.cuda()

    # predict action
    preds, _ = model(states, timesteps, actions=actions, key_state_mask=key_state_mask)

    return preds[:, -1]


def update(model, action_hist, state_hist, actions, states, t):
    # A function used to update the state and action history.
    actions = torch.from_numpy(actions)
    if len(state_hist) == model.key_net.block_size // 2:  # The context buffer is full.
        assert len(action_hist) == model.key_net.block_size // 2 - 1
        state_hist = state_hist[1:] + [states]
        action_hist = action_hist[1:] + [actions]
        t += 1
    else:
        state_hist.append(states)
        action_hist.append(actions)
    return action_hist, state_hist, t


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

    parser.add_argument("--eval_max_steps", default=200, type=int, help="Max steps allowed in eval.")

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
    state_dim = state_dict_from_ckpt['key_net.state_encoder.net.0.weight'].shape[1]
    action_dim = state_dict_from_ckpt['key_net.action_encoder.net.0.weight'].shape[1]
    max_timestep = state_dict_from_ckpt['key_net.global_pos_emb'].shape[1]
    print('Loaded ckpt from:', path)

    # Load demos to fetch the env. seeds used in training.
    traj_path = os.path.join(
        DATA_PATH,
        f'{args.task}/trajectory.{args.obs_mode}.{args.control_mode}.h5')
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

    dataset['env_states'] = [
        np.array(
            traj_all[f"traj_{i}"]['env_states']
        ) for i in ids
    ]

    dataset['obs'] = [
        np.array(
            traj_all[f"traj_{i}"]["obs"]
        ) for i in ids
    ]

    dataset['actions'] = [
        np.array(
            traj_all[f"traj_{i}"]["actions"]
        ) for i in ids
    ]

    print(dataset['env_states'][0].shape[0])

    dataset['key_label'] = [
        [[] for j in range(dataset['env_states'][idx].shape[0])] for idx in range(length)
    ]
    print(dataset['key_label'][0])

    max_steps = np.max(len(s) for s in dataset['env_states'])

    print(dataset['env_states'][0].shape, type(dataset['env_states'][0]))
    print(dataset['obs'][0].shape, type(dataset['obs'][0]))
    print(dataset['actions'][0].shape, type(dataset['actions'][0]))

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
            # print(f'traj_{idx}')
            for step_idx, key in enumerate(dataset['infos/is_contacted'][idx]):
                # print(f'{step_idx}\t{key}')
                dataset['key_label'][idx][step_idx].append('is_contacted')
                if key: break

    # If PegInsertion (three key states)
    # key state I: is_grasped -> true
    # key state II: pre_inserted -> true
    # key state III: end of the trajectory
    if args.task == 'PegInsertionSide-v0':
        for idx in range(length):
            # print(f'traj_{idx}')
            for step_idx, key in enumerate(dataset['infos/is_grasped'][idx]):
                # print(f'{step_idx}\t{key}')
                dataset['key_label'][idx][step_idx].append('is_grasped')
                if key: break
            for step_idx, key in enumerate(dataset['infos/pre_inserted'][idx]):
                # print(f'{step_idx}\t{key}')
                dataset['key_label'][idx][step_idx].append('pre_inserted')
                if key: break
            for step_idx in range(dataset['env_states'][idx].shape[0]):
                dataset['key_label'][idx][step_idx].append('end')

    # print(dataset['key_label'][0], type(dataset['key_label'][0]))

    for l in dataset['key_label'][0]:
        print(l)
    #     for i in l:
    #         print(i)

    # config our net
    # key_config = KeyNetConfig(
    #     block_size=params['context_length'],
    #     n_layer=params['n_key_layer'],
    #     n_embd=params['n_embd'],
    #     n_head=params['n_head'],
    #     model_type=params['keynet_type'],
    #     attn_pdrop=float(params['dropout']),
    #     resid_pdrop=float(params['dropout']),
    #     embd_pdrop=float(params['dropout']),
    #     max_timestep=max_timestep,
    # )
    # act_config = ActNetConfig(
    #     block_size=params['context_length'],
    #     n_layer=params['n_act_layer'],
    #     n_embd=params['n_embd'],
    #     n_head=params['n_head'],
    #     model_type=params['actnet_type'],
    #     attn_pdrop=float(params['dropout']),
    #     resid_pdrop=float(params['dropout']),
    #     key_states=params['key_states']
    # )
    # autocot_model = AutoCoT(
    #     key_config=key_config,
    #     vq_len=params['vq_len'],
    #     vq_beta=float(params['vq_beta']),
    #     vq_legacy=params['vq_legacy'],
    #     vq_log=params['vq_log'],
    #     act_config=act_config,
    #     optimizers_config=None,
    #     scheduler_config=None,
    #     state_dim=state_dim,
    #     action_dim=action_dim
    # )
    # autocot_model = autocot_model.cuda()
    # autocot_model.load_state_dict(state_dict_from_ckpt, strict=False)
    # autocot_model.eval()
    #
    # output_str, output_dict = '', dict()
    #
    # metric_dict = defaultdict(lambda: [[] for _ in range(len(eval_ids))])
    # for start_idx in tqdm(range(0, len(eval_ids), n_env)):
    #     reset_args_list = []
    #     for i in range(start_idx, min(start_idx + n_env, len(eval_ids))):
    #         reset_kwargs = json_data["episodes"][eval_ids[i]]['reset_kwargs']
    #         reset_args_list.append(reset_kwargs)
    #
    #     s = torch.from_numpy(envs.reset(reset_args_list)).float()
    #     state_hist, action_hist, t = [s], [], np.zeros([n_env])
    #
    #     for step in range(args.eval_max_steps):
    #         a = predict(autocot_model, action_hist, state_hist, t).cpu().numpy()
    #
    #         s, _, _, infos = envs.step(a)
    #         s = torch.from_numpy(s).float()
    #
    #         action_hist, state_hist, t = update(
    #             autocot_model, action_hist, state_hist, a, s, t)
    #
    #         # Update metrics.
    #
    #         for i, info in enumerate(infos):
    #             j = start_idx + i
    #             # print(info.keys())
    #             # You might want to use these additional metrics.
    #             '''
    #             if args.task == 'PickCube-v0':
    #                 metric_dict['is_grasped'][j].append(info['is_grasped'])
    #             if args.task == 'StackCube-v0':
    #                 metric_dict['is_cubaA_grasped'][j].append(info['is_cubaA_grasped'])
    #                 metric_dict['is_cubeA_on_cubeB'][j].append(info['is_cubeA_on_cubeB'])
    #             if args.task == 'PegInsertionSide-v0':
    #                 metric_dict['is_grasped'][j].append(info['is_grasped'])
    #                 metric_dict['pre_inserted'][j].append(info['pre_inserted'])
    #             if args.task == 'TurnFaucet-v0':
    #                 metric_dict['is_contacted'][j].append(info['is_contacted'])
    #             if args.task == 'PushChair-v1':
    #                 metric_dict['close_to_target'][j].append(info['chair_close_to_target'])
    #                 metric_dict['static_at_last'][j].append(
    #                     info['chair_close_to_target'] and info['chair_static'])
    #             '''
    #             metric_dict['success'][j].append(info['success'])
    #
    # for k, v in metric_dict.items():
    #     v = np.mean([np.any(vv) for vv in v]) * 100
    #     output_str += f'{k} {v:.2f}, '
    #     output_dict[k] = v
    # output_str = output_str[:-2]
    # print(output_str)


