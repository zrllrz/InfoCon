import os
import numpy as np
import h5py

from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

# Please specify the DATA_PATH (the base folder for storing data) in `path.py`.
from path import DATA_PATH


class MS2Demos(Dataset):
    def __init__(self,
                 data_split='train',
                 task='PickCube-v0',
                 obs_mode='state',
                 control_mode='pd_joint_delta_pos',
                 length=-1,
                 min_seq_length=None,
                 max_seq_length=None,
                 with_key_states=False,
                 multiplier=20,  # Used for faster data loading.
                 seed=None):  # seed for train/test spliting.
        super().__init__()
        self.task = task
        self.data_split = data_split
        self.seed = seed
        self.min_seq_length = min_seq_length  # For sampling trajectories.
        self.max_seq_length = max_seq_length  # For sampling trajectories.
        self.with_key_states = with_key_states  # Whether output key states.
        self.multiplier = multiplier

        # Usually set min and max traj length to be the same value.
        self.max_steps = -1  # Maximum timesteps across all trajectories.
        traj_path = os.path.join(DATA_PATH,
                                 f'{task}/trajectory.{obs_mode}.{control_mode}.h5')
        key_path = os.path.join(DATA_PATH, f'{task}/keys-0907.txt')
        print('Traj path:', traj_path)
        print('Key path:', key_path)
        self.data = self.load_demo_dataset(traj_path, key_path, length)

        # Cache key states for faster data loading.
        if self.with_key_states:
            self.idx_to_key_states = dict()

    def __len__(self):
        return len(self.data['env_states'])

    def __getitem__(self, index):
        # Offset by one since the last obs does not have a corresponding action.
        l = len(self.data['obs'][index]) - 1

        # Sample starting and ending index given the min and max traj length.
        if self.min_seq_length is None and self.max_seq_length is None:
            s_idx, e_idx = 0, l
        else:
            min_length = 0 if self.min_seq_length is None else self.min_seq_length
            max_length = l if self.max_seq_length is None else self.max_seq_length
            assert min_length <= max_length
            if min_length == max_length:
                length = min_length
            else:
                length = np.random.randint(min_length, max_length, 1)[0]
            if length <= l:
                s_idx = np.random.randint(0, l - length + 1, 1)[0]
                e_idx = s_idx + length
            else:
                s_idx, e_idx = 0, l
        assert e_idx <= l, f'{e_idx}, {l}'

        # Call get_key_states() if you want to use the key states.
        # Here `s` is the state observation, `a` is the action,
        # `env_states` not used during training (can be used to reconstruct env for debugging).
        # `t` is used for positional embedding as in Decision Transformer.
        data_dict = {
            's': self.data['obs'][index][s_idx:e_idx].astype(np.float32),
            'a': self.data['actions'][index][s_idx:e_idx].astype(np.float32),
            't': np.array([s_idx]).astype(np.float32),
            # 'env_states': self.data['env_states'][index][s_idx:e_idx].astype(np.float32),
        }
        if self.with_key_states:
            if f'key_states_{index}' not in self.idx_to_key_states:
                self.idx_to_key_states[f'key_states_{index}'] = self.get_key_states(index)
            data_dict['k'] = self.idx_to_key_states[f'key_states_{index}'][0]
            data_dict['km'] = self.idx_to_key_states[f'key_states_{index}'][1]
        return data_dict

    def info(self):  # Get observation and action shapes.
        return self.data['obs'][0].shape[-1], self.data['actions'][0].shape[-1]

    def load_demo_dataset(self, path, key_path, length):
        dataset = {}
        traj_all = h5py.File(path)
        if length == -1:
            length = len(traj_all)
        np.random.seed(self.seed)  # Fix the random seed for train/test data split.

        # Since TurnFaucet uses 10 different faucet models, we shuffle the data
        # such that the resulting sampled data are evenly sampled across faucet models.
        if self.task == 'TurnFaucet-v0':
            ids = []
            for i in range(10):  # Hard-code the 10 data splits for permutation.
                t_ids = np.random.permutation(len(traj_all) // 10)[:length // 10]
                t_ids += i * len(traj_all) // 10
                ids.append(t_ids)
            ids = np.concatenate(ids)
        # Since PushChair uses 5 different faucet models, we shuffle the data
        # such that the resulting sampled data are evenly sampled across chair models.
        elif self.task == 'PushChair-v1':
            ids = []
            for i in range(5):  # Hard-code the 5 data splits for permutation.
                t_ids = np.random.permutation(len(traj_all) // 5)[:length // 5]
                t_ids += i * len(traj_all) // 5
                ids.append(t_ids)
            ids = np.concatenate(ids)
        else:
            ids = np.random.permutation(len(traj_all))[:length]

        ids = ids.tolist() * self.multiplier  # Duplicate the data for faster loading.

        # Note that the size of `env_states` and `obs` is that of the others + 1.
        # And most `infos` is for the next obs rather than the current obs.

        # `env_states` is used for reseting the env (might be helpful for eval)
        dataset['env_states'] = [np.array(
            traj_all[f"traj_{i}"]['env_states']) for i in ids]
        # `obs` is the observation of each step.
        dataset['obs'] = [np.array(traj_all[f"traj_{i}"]["obs"]) for i in ids]
        dataset['actions'] = [np.array(traj_all[f"traj_{i}"]["actions"]) for i in ids]

        # actions = np.concatenate(dataset['actions'])
        # actions_std = np.std(actions, 0)
        # dataset['actions'] = [
        #    np.array(traj_all[f"traj_{i}"]["actions"]) / (actions_std + 1e-7) for i in ids]

        # `rewards` is not currently used in CoTPC training.
        dataset['rewards'] = [np.array(traj_all[f"traj_{i}"]["rewards"]) for i in ids]

        # Here we can read the key states from the txt we made (I do not like json ... sorry :) )
        dataset['key_state_step'] = list()
        with open(key_path, 'r') as fk:
            lines = fk.readlines()  # Read all lines
            for i in ids:
                line = lines[i].split(',')[:-1]
                line = np.array([int(item) for item in line])
                dataset['key_state_step'].append(line)

        self.max_steps = np.max([len(s) for s in dataset['env_states']])

        return dataset

    def get_key_states(self, idx):
        # Note that `infos` is for the next obs rather than the current obs.
        # Thus, we need to offset the `step_idx`` by one.

        key_state_step = self.data['key_state_step'][idx]
        # print('key_state_step =', key_state_step)

        key_states = [self.data['obs'][idx][step] for step in key_state_step]
        # print('key_states', key_states)

        key_state_mask = np.array([1.0 * (step != -1) for step in key_state_step])
        # print('key_state_mask', key_states)

        key_states = np.stack(key_states, 0).astype(np.float32)
        assert len(key_states) > 0, self.task
        return key_states, key_state_mask


# To obtain the padding function for sequences.
def get_padding_fn(data_names):
    assert 's' in data_names, 'Should at least include `s` in data_names.'

    def pad_collate(*args):
        assert len(args) == 1
        output = {k: [] for k in data_names}
        for b in args[0]:  # Batches
            for k in data_names:
                output[k].append(torch.from_numpy(b[k]))

        # Include the actual length of each sequence sampled from a trajectory.
        # If we set max_seq_length=min_seq_length, this is a constant across samples.
        output['lengths'] = torch.tensor([len(s) for s in output['s']])

        # Padding all the sequences.
        for k in data_names:
            output[k] = pad_sequence(output[k], batch_first=True, padding_value=0)

        return output

    return pad_collate


# Sample code for the data loader.
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # The default values for CoTPC for tasks in ManiSkill2.
    batch_size, num_traj, seed, min_seq_length, max_seq_length, task = \
        256, 500, 0, 60, 60, 'PickCube-v0'
    batch_size, num_traj, seed, min_seq_length, max_seq_length, task = \
        256, 500, 0, 60, 60, 'PushChair-v1'

    train_dataset = MS2Demos(
        # control_mode='pd_joint_delta_pos',
        control_mode='base_pd_joint_vel_arm_pd_joint_vel',
        length=num_traj, seed=seed,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        with_key_states=True,
        task=task)

    collate_fn = get_padding_fn(['s', 'a', 't', 'k', 'km'])
    train_data = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn)

    data_iter = iter(train_data)
    data = next(data_iter)
    # print(len(data))  # 4
    # for k, v in data.items():
    #     print(k, v.shape)
    # 's', [256, 60, 51]
    # 'a', [256, 60, 8]
    # 't', [256, 1]
    # 'k', [256, 2, 51]
