import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl

from data import MS2Demos, get_padding_fn
from autocot import (
    KeyNetConfig,
    ActCommitNetConfig,
    AutoCoT
)
from lr_scheduler import CosineAnnealingLRWarmup
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks import MySaveLogger

from path import MODEL_PATH

'''
Parsing Commandline Input
'''
def parse_args():
    parser = argparse.ArgumentParser()
    # Training hyper-parameters.
    parser.add_argument("--n_iters", default=4_800_000, type=int, help="Number of training iterations.")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
    parser.add_argument("--init_lr", default='5e-4', type=str, help="The initial learning rate.")
    parser.add_argument("--weight_decay", default='0', type=str, help="Weight decay coefficient.")
    parser.add_argument("--beta1", default='0.9', type=str, help="Beta1 in the Adam optimizer.")
    parser.add_argument("--beta2", default='0.95', type=str, help="Beta2 in the Adam optimizer.")
    parser.add_argument("--dropout", default='0.0', type=str, help="Dropout probability.")
    parser.add_argument("--lr_schedule", default='cos_decay_with_warmup', type=str,
                        help="The learning rate schedule.")
    parser.add_argument("--t_warmup", default=1000, type=int,
                        help="(Make sure you're using \'CosineAnnealingLRWarmup\') Warm-up iteration steps")
    parser.add_argument("--milestones", default=[78000], type=list,
                        help="(Make sure you're using \'MultiStepLR\') Time steps to decay lr")
    parser.add_argument("--gamma", default='0.1', type=str,
                        help="(Make sure you're using \'MultiStepLR\') Decay of lr after each milestone step")

    # General hyper-parameters for the GPT architecture.
    parser.add_argument("--n_head", default=8, type=int, help="Number of attention heads.")
    parser.add_argument("--n_embd", default=128, type=int, help="Hidden feature dimension.")

    # Hyper-parameters regarding key_net, key_book, act_net, commit_net
    parser.add_argument("--n_key_layer", default=4, type=int,
                        help="Number of attention layers in KeyNet")

    parser.add_argument('--vq_n_e', type=int, default=100,
                        help="How many kinds of keys in the key_book")
    parser.add_argument('--vq_beta', type=str, default='0.2',
                        help="Coefficient in the VQ loss")
    parser.add_argument('--vq_legacy', type=bool, default=False,
                        help="Place that add vq_beta, should always be False")
    parser.add_argument('--vq_log', type=bool, default=True,
                        help="log variation of indices choice")
    parser.add_argument('--vq_kmeans_reset', type=int, default=None,
                        help="steps interval of resetting embedding using k-means")
    parser.add_argument('--vq_kmeans_step', type=int, default=None,
                        help="interation steps of k-means")
    parser.add_argument("--n_act_layer", default=4, type=int,
                        help="Number of attention layers in ActNet")
    parser.add_argument("--n_commit_layer", default=4, type=int,
                        help="Number of attention layers in commit_net")
    parser.add_argument("--subgoal_marginal", default='1e-6', type=str,
                        help="Triple loss marginal for sub-goal loss")

    # General hyper-parameters regarding module loading and saving
    parser.add_argument("--model_name", default='TEST', type=str, help="Model name (for storing ckpts).")
    parser.add_argument("--from_model_name", default='', type=str, help="Name of the pretrained module.")
    parser.add_argument("--from_ckpt", default=-1, type=int, help="Ckpt of pretrained module.")

    # Hyper-parameters regarding the demo dataset
    parser.add_argument('--task', type=str, default='PegInsertionSide-v0', help="Task (env-id) in ManiSkill2.")
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
    parser.add_argument("--save_every", default=10, type=int, help="Save module every # epoch.")
    parser.add_argument("--log_every", default=10, type=int, help="log metrics every # iters.")

    # For faster data loader.
    parser.add_argument("--num_workers", default=5, type=int,
                        help="A positive number for fast async data loading.")
    parser.add_argument('--multiplier', type=int, default=52,
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
        n_embd=args.n_embd,
        n_head=args.n_head,
        attn_pdrop=float(args.dropout),
        resid_pdrop=float(args.dropout),
        embd_pdrop=float(args.dropout),
        block_size=args.context_length,
        n_layer=args.n_key_layer,
        max_timestep=train_dataset.max_steps,
    )
    act_config = ActCommitNetConfig(
        n_embd=args.n_embd,
        n_head=args.n_head,
        attn_pdrop=float(args.dropout),
        resid_pdrop=float(args.dropout),
        embd_pdrop=float(args.dropout),
        block_size=args.context_length,
        n_layer=args.n_act_layer,
        max_timestep=train_dataset.max_steps,
        commit=False
    )
    commit_config = ActCommitNetConfig(
        n_embd=args.n_embd,
        n_head=args.n_head,
        attn_pdrop=float(args.dropout),
        resid_pdrop=float(args.dropout),
        embd_pdrop=float(args.dropout),
        block_size=args.context_length,
        n_layer=args.n_commit_layer,
        max_timestep=train_dataset.max_steps,
        commit=True
    )

    optimizer_config = {
        'init_lr': float(args.init_lr),
        'weight_decay': float(args.weight_decay),
        'beta1': float(args.beta1),
        'beta2': float(args.beta2),
    }
    assert args.lr_schedule in ['cos_decay_with_warmup', 'multistep', None], 'Unknown lr scheduler'
    if args.lr_schedule == 'cos_decay_with_warmup':
        scheduler_config = {
            'type': 'cos_decay_with_warmup',
            't_max': args.n_iters,
            't_warmup': args.t_warmup
        }
    elif args.lr_schedule == 'multistep':
        scheduler_config = {
            'type': 'multistep',
            'milestones': args.milestones,
            'gamma': float(args.gamma)
        }
    else:
        scheduler_config = None

    autocot_model = AutoCoT(
        key_config=key_config,
        vq_n_e=args.vq_n_e,
        vq_beta=float(args.vq_beta),
        vq_legacy=args.vq_legacy,
        vq_log=args.vq_log,
        vq_kmeans_reset=args.vq_kmeans_reset,
        vq_kmeans_step=args.vq_kmeans_step,
        act_config=act_config,
        commit_config=commit_config,
        subgoal_marginal=float(args.subgoal_marginal),
        optimizers_config=optimizer_config,
        scheduler_config=scheduler_config,
        state_dim=state_dim,
        action_dim=action_dim,
        key_dim=args.n_embd
    )

    # autocot_model.configure_optimizers()
    model_path = os.path.join(MODEL_PATH, args.model_name)
    os.makedirs(model_path, exist_ok=True)
    # If loaded from pretrained module first.
    if args.from_ckpt > 0:
        if args.from_model_name:
            path = os.path.join(
                MODEL_PATH, args.from_model_name, f'{args.from_ckpt}.pth')
        else:
            path = os.path.join(model_path, f'{args.from_ckpt}.pth')
        autocot_model.load_state_dict(torch.load(path), strict=True)
        print(f'Pretrained module loaded from {path}.')

    log_path = os.path.join(model_path, 'log.txt')

    mysavelogger = MySaveLogger(
        path=model_path,
        args=args,
        epoch_frequency=args.save_every
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[mysavelogger],
        max_steps=args.n_iters,
    )
    trainer.fit(
        model=autocot_model,
        train_dataloaders=train_data,
    )

