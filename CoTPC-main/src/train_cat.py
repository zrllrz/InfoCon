import os
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from collections import deque
from tqdm import tqdm
import numpy as np

from data_autocot_cat import MS2Demos, get_padding_fn
from model_cat import GPTConfig, GPTCat
from train_utils import CosineAnnealingLRWarmup

try:
    # Use might need this for wandb to work due to protobuf issues.
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

    import wandb

    USE_WANDB = True
    PROJECT_NAME = 'CoTPC'  # Please specify the project name.
except ImportError:
    print('Do not use wandb since it is not found.')
    USE_WANDB = False

# Please specify MODEL_PATH (the base folder for storing models) in `path.py`.
from path import MODEL_PATH


def parse_args():
    parser = argparse.ArgumentParser()

    # Training hyper-parameters.
    parser.add_argument("--n_iters", default=1_600_000, type=int, help="Number of training iterations.")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
    parser.add_argument("--init_lr", default='5e-4', type=str, help="The initial learning rate.")
    parser.add_argument("--weight_decay", default='0', type=str, help="Weight decay coefficient.")
    parser.add_argument("--beta1", default='0.9', type=str, help="Beta1 in the Adam optimizer.")
    parser.add_argument("--beta2", default='0.95', type=str, help="Beta2 in the Adam optimizer.")
    parser.add_argument("--dropout", default='0.0', type=str, help="Dropout probability.")
    parser.add_argument("--lr_schedule", default='cos_decay_with_warmup', type=str,
                        help="The learning rate schedule.")

    # Hyper-parameters regarding CoTPC.
    parser.add_argument("--key_state_coeff", default=0.0, type=float,
                        help="Coefficient for the key state prediction loss.")
    parser.add_argument('--model_type', type=str, default='s+a+k',
                        help="Model type for the CoTPC model (see GPTConfig).")

    # General hyper-parameters regarding model loading and saving
    parser.add_argument("--model_name", default='', type=str, help="Model name (for storing ckpts).")
    parser.add_argument("--from_model_name", default='', type=str, help="Name of the pretrained model.")
    parser.add_argument("--from_ckpt", default=-1, type=int, help="Ckpt of pretrained model.")

    # Hyper-parameters regarding the demo dataset
    parser.add_argument('--task', type=str, default='PickCube-v0', help="Task (env-id) in ManiSkill2.")
    parser.add_argument('--control_mode', type=str, default='pd_joint_delta_pos',
                        help="Control mode used in envs from ManiSkill2.")
    parser.add_argument('--obs_mode', type=str, default='state',
                        help="State mode used in envs from ManiSkill2.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for data spliting.")
    parser.add_argument("--num_traj", default=-1, type=int, help="Number of training trajectories.")
    parser.add_argument('--context_length', type=int, default=60,
                        help="Context size of CoTPC (the maximium length of sequences " +
                             "sampled from demo trajectories in training).")
    parser.add_argument('--min_seq_length', type=int, default=60,
                        help="Mininum length of sequences sampled from demo trajectories in training.")

    # Save and log frequencies.
    parser.add_argument("--save_every", default=40000, type=int, help="Save model every # iters.")
    parser.add_argument("--log_every", default=2000, type=int, help="log metrics every # iters.")

    # General hyper-parameters for the GPT architecture.
    parser.add_argument("--n_layer", default=4, type=int, help="Number of attention layers.")
    parser.add_argument("--n_head", default=8, type=int, help="Number of attention heads.")
    parser.add_argument("--n_embd", default=128, type=int, help="Hidden feature dimension.")

    # For faster data loader.
    parser.add_argument("--num_workers", default=2, type=int,
                        help="A positive number for fast async data loading.")
    parser.add_argument('--multiplier', type=int, default=20,
                        help="Duplicate the dataset to reduce data loader overhead.")

    parser.add_argument('--keys_name', type=str, default="keys.txt",
                        help="Duplicate the dataset to reduce data loader overhead.")

    return parser.parse_args()


def mse_loss_with_weights(preds, targets, weights=None):
    losses = torch.mean((preds - targets) ** 2, -1)
    if weights is None:
        return torch.mean(losses)
    else:
        assert losses.shape == weights.shape, losses.shape
        return torch.mean(losses * weights)


def get_loss(preds, targets, lengths):
    # If we have sequences of varied lengths, use masks so we do not compute loss
    # over padded values. If we set max_seq_length=min_seq_length, then it should
    # not matter since all sequences have the same length.
    B = preds.shape[0]
    max_len = torch.max(lengths)  # Max length of the current mini-batch.
    lengths = lengths[:, None]  # B x 1
    temp = torch.arange(0, max_len)[None].expand(B, -1).cuda()  # B x max_len
    masks = (temp < lengths.expand(B, max_len)).float()  # B x max_len

    loss = mse_loss_with_weights(
        preds.reshape(-1, preds.size(-1)),
        targets.reshape(-1, targets.size(-1)),
        masks.reshape(-1))
    return loss


if __name__ == "__main__":

    args = parse_args()
    assert args.model_name != '', 'Should specify --model_name'
    print('Model name:', args.model_name)

    train_dataset = MS2Demos(
        task=args.task,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        length=args.num_traj,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.context_length,
        with_key_states='k' in args.model_type,
        multiplier=args.multiplier,
        seed=args.seed,
        keys_name=args.keys_name
    )
    print('Training data size:', len(train_dataset))
    print('Max steps:', train_dataset.max_steps)

    input_dict = ['s', 'a', 't']
    input_dict += ['k'] if 'k' in args.model_type else []
    collate_fn = get_padding_fn(input_dict)
    train_data = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,  # Faster data loading if using GPU.
        num_workers=args.num_workers,
        persistent_workers=True,  # Faster data loader resets.
        collate_fn=collate_fn,
        drop_last=True,
    )
    data_iter = iter(train_data)

    state_dim, action_dim = train_dataset.info()

    conf = GPTConfig(
        args.context_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        model_type=args.model_type,
        max_timestep=train_dataset.max_steps,
        embd_pdrop=float(args.dropout),
        resid_pdrop=float(args.dropout),
        attn_pdrop=float(args.dropout),
    )
    print('state_dim', state_dim)
    model = GPTCat(conf, state_dim=state_dim, action_dim=action_dim).cuda()
    optimizer = model.configure_adamw_optimizers({
        'init_lr': float(args.init_lr),
        'weight_decay': float(args.weight_decay),
        'beta1': float(args.beta1),
        'beta2': float(args.beta2),
    })

    # Learning rate schedules (which might require more tuning).
    if args.lr_schedule == 'cos_decay_with_warmup':
        lr_scheduler = CosineAnnealingLRWarmup(
            optimizer, T_max=args.n_iters, T_warmup=1000)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[78000], gamma=0.1)

    model_path = os.path.join(MODEL_PATH, args.model_name)
    os.makedirs(model_path, exist_ok=True)

    # If loaded from pretrained model first.
    if args.from_ckpt > 0:
        if args.from_model_name:
            path = os.path.join(
                MODEL_PATH, args.from_model_name, f'{args.from_ckpt}.pth')
        else:
            path = os.path.join(model_path, f'{args.from_ckpt}.pth')
        model.load_state_dict(torch.load(path), strict=True)
        print(f'Pretrained model loaded from {path}.')

    log_path = os.path.join(model_path, 'log.txt')
    if USE_WANDB:
        wandb.init(
            project=PROJECT_NAME, name=args.model_name, config=args,
            config_exclude_keys=['model_name', 'save_every', 'log_every'],
        )
        wandb.run.log_code(".")  # Need to first enable it on wandb web UI.

    losses_act_pred = deque(maxlen=1000)
    losses_key_states = deque(maxlen=1000)

    # Main training loop.
    for idx in tqdm(range(args.n_iters + 1)):

        # Adjust lr schedule when loaded from pretrained models.
        if args.from_ckpt > 0 and idx <= args.from_ckpt:
            lr_scheduler.step()
            continue

        # Obtain the current mini-batch (infinite loop).
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_data)
            batch = next(data_iter)
        batch = {k: v.cuda() for k, v in batch.items()}

        # Forward pass.
        a_preds, k_preds = model(batch['s'], batch['t'], batch['a'], batch['k'])

        # Obtain training losses.
        loss_a_preds = get_loss(a_preds, batch['a'], batch['lengths'])
        loss_k_preds = get_loss(k_preds, batch['k'], batch['lengths'])
        total_loss = loss_a_preds + args.key_state_coeff * loss_k_preds

        losses_act_pred.append(loss_a_preds.item())
        losses_key_states.append(loss_k_preds.item())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if idx % args.log_every == 0:
            with open(log_path, 'a' if os.path.exists(log_path) else 'w') as f:
                avg_loss_act_pred = np.mean(losses_act_pred)
                avg_loss_key_states = np.mean(losses_key_states)
                print(f'Iteration {idx}: {avg_loss_act_pred}, {avg_loss_key_states}')
                f.write(f'{idx},{avg_loss_act_pred},{avg_loss_key_states}\n')
                if USE_WANDB:
                    log_dict = {
                        'n_iter': idx,
                        'loss_actions': avg_loss_act_pred,
                        'loss_sum': avg_loss_act_pred,
                    }
                    if 'cot' in args.model_type:
                        log_dict['loss_key_states'] = avg_loss_key_states
                        log_dict['loss_sum'] = avg_loss_act_pred + avg_loss_key_states
                    wandb.log(log_dict)

        if idx > 0 and idx % args.save_every == 0:
            save_path = os.path.join(model_path, f'{idx}.pth')
            torch.save({
                'model': model.state_dict(),
                'metadata': vars(args)
            }, save_path)

        # Update learning rate.
        lr_scheduler.step()
