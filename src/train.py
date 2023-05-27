import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl

from .data import MS2Demos, get_padding_fn
from .autocot import (
    KeyNetConfig,
    ActNetConfig,
    AutoCoT,
)
from .lr_scheduler import CosineAnnealingLRWarmup

from path import MODEL_PATH

'''
Parsing Commandline Input
'''
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

    # General hyper-parameters for the GPT architecture.
    parser.add_argument("--n_head", default=8, type=int, help="Number of attention heads.")
    parser.add_argument("--n_embd", default=128, type=int, help="Hidden feature dimension.")
    # Hyper-parameters regarding KeyNet, KeyBook, ActNet
    parser.add_argument('--keynet_type', type=str, default='s+a',
                        help="Model type for the KeyNet module")
    parser.add_argument("--n_key_layer", default=3, type=int,
                        help="Number of attention layers in KeyNet")
    parser.add_argument('--len_book', type=int, default=10,
                        help="Length of the Key States Codebook")
    parser.add_argument('--vq_beta', type=float, default=2.0,
                        help="Coefficient in the VQ loss")
    parser.add_argument('--vq_legacy', type=bool, default=False,
                        help="Place that add vq_beta, should always be False")
    parser.add_argument('--actnet_type', type=str, default='s+a+cot',
                        help="Model type for the ActNet module")
    parser.add_argument("--n_act_layer", default=1, type=int,
                        help="Number of attention layers in ActNet")
    parser.add_argument('--key_states', type=str, default='a',
                        help="Which key states to use (see GPTConfig for the spec. format).")

    # General hyper-parameters regarding module loading and saving
    parser.add_argument("--model_name", default='', type=str, help="Model name (for storing ckpts).")
    parser.add_argument("--from_model_name", default='', type=str, help="Name of the pretrained module.")
    parser.add_argument("--from_ckpt", default=-1, type=int, help="Ckpt of pretrained module.")

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
    parser.add_argument("--save_every", default=40000, type=int, help="Save module every # iters.")
    parser.add_argument("--log_every", default=2000, type=int, help="log metrics every # iters.")

    # For faster data loader.
    parser.add_argument("--num_workers", default=2, type=int,
                        help="A positive number for fast async data loading.")
    parser.add_argument('--multiplier', type=int, default=20,
                        help="Duplicate the dataset to reduce data loader overhead.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.model_name != '', 'Should specify --model_name'
    print('Model name:', args.model_name)

    print('Preparing Training Data...')
    train_dataset = MS2Demos(
        control_mode=args.control_mode,
        obs_mode=args.obs_mode,
        length=args.num_traj, seed=args.seed,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.context_length,
        with_key_states=True,
        task=args.task, multiplier=args.multiplier)
    print('Training data size:', len(train_dataset))
    print('Max steps:', train_dataset.max_steps)
    collate_fn = get_padding_fn(['s', 'a', 't', 'k'])
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

    state_dim, action_dim = train_dataset.info()
    key_config = KeyNetConfig(
        block_size=args.context_length,
        n_embd=args.n_embd,
        n_head=args.n_head,
        model_type=args.keynet_type,
        attn_pdrop=float(args.dropout),
        resid_pdrop=float(args.dropout),
        embd_pdrop=float(args.dropout),
        max_timestep=train_dataset.max_steps,
    )
    act_config = ActNetConfig(
        block_size=args.context_length,
        n_embd=args.n_embd,
        n_head=args.n_head,
        model_type=args.actnet_type,
        attn_pdrop=float(args.dropout),
        resid_pdrop=float(args.dropout),
        key_states=args.key_states
    )
    optimizer_config = {
        'init_lr': args.init_lr,
        'weight_decay': args.weight_decay,
        'beta1': args.beta1,
        'beta2': args.beta2,
    }
    autocot_model = AutoCoT(
        key_config=key_config,
        len_book=args.len_book,
        vq_beta=args.vq_beta,
        vq_legacy=args.vq_legacy,
        act_config=act_config,
        optimizers_config=optimizer_config,
        state_dim=state_dim,
        action_dim=action_dim
    )

