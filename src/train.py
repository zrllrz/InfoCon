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
    ImplicitSAResFCConfig,
    ImplicitSAGPTConfig,
    ExplicitSAGPTConfig,
    ExplicitSAHNGPTConfig,
    ExplicitSAHNConfig,
    ActCommitNetConfig,
    RecNetConfig,
    ENetConfig,
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
    parser.add_argument("--dim_key", default=128, type=int, help="Hidden feature dimension.")
    parser.add_argument("--dim_e", default=1024, type=int, help="Hidden feature dimension.")

    # Hyper-parameters regarding key_net, key_book, act_net, commit_net
    # parser.add_argument("--n_rec_layer", default=4, type=int,
    #                     help="Number of attention layers in RecNet")
    parser.add_argument("--n_key_layer", default=4, type=int,
                        help="Number of attention layers in KeyNet")
    parser.add_argument("--te_key_dim", default=128, type=int,
                        help="Number of attention layers in KeyNet")
    parser.add_argument("--n_rec_layer", default=4, type=int,
                        help="Number of attention layers in RecNet")

    parser.add_argument('--vq_n_e', type=int, default=100,
                        help="How many kinds of keys in the key_book")
    parser.add_argument('--KT', type=str, default='1.0',
                        help="Temperature for classifier")
    parser.add_argument('--coe_lip', type=str, default='2.0',
                        help="Lip Constant")
    parser.add_argument("--coe_cluster", default='0.1', type=str, help="cluster weight")
    parser.add_argument("--coe_rec", default='1.0', type=str, help="reconstruction weight from key_soft")
    # parser.add_argument('--repulse', action='store_true',
    #                     help="weight of clustering (classifier) loss")
    # parser.add_argument('--c_ss', type=str, default='1.0',
    #                     help="weight of clustering soft-soft loss")
    # parser.add_argument('--c_sh', type=str, default='0.01',
    #                     help="weight of clustering soft-hard loss")
    # parser.add_argument('--c_hs', type=str, default='0.1',
    #                     help="weight of clustering hard-soft loss")
    # parser.add_argument('--c_hh', type=str, default='0.001',
    #                     help="weight of clustering hard-hard loss")
    # parser.add_argument('--vq_decay_energy', type=str, default='0.1', help="decay coe for energy calc")
    # parser.add_argument('--vq_coe_structure', type=str, default='0.1', help="Coefficient in the VQ loss")

    parser.add_argument("--sa_type", default='gpt', type=str, choices=['gpt', 'egpt', 'egpthn', 'resfc', 'hn'],
                        help="type of sa_net")
    parser.add_argument("--n_state_layer", default=1, type=int,
                        help="Number of layers for state prediction in SANet")
    parser.add_argument("--n_action_layer", default=1, type=int,
                        help="Number of layers (after state prediction) for action prediction in SANet")
    parser.add_argument("--use_pos_emb", action='store_true',
                        help="if True, use key energy gradient to evaluate effect of key states, only use when resfc")
    parser.add_argument("--use_skip", action='store_true',
                        help="if True, use skip connection for HN generated net when using HN")

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
    # assert args.model_name != '', 'Should specify --model_name'
    print(args)

    # str_repulse = 'attract_and_repulse' if args.repulse else 'attract_only'

    auto_model_name = \
        args.model_name \
        + 'k' + str(args.n_key_layer) \
        + '-r' + str(args.n_rec_layer) \
        + '-c' + str(args.vq_n_e) \
        + '_KT' + args.KT + '_LIP' + args.coe_lip \
        + '-' + args.sa_type + '_s' + str(args.n_state_layer) + '_a' + str(args.n_action_layer) \
        + '-emb' + str(args.n_embd) \
        + '-key' + str(args.dim_key) \
        + '-e' + str(args.dim_e) \
        + '-cluster' + args.coe_cluster + '-rec' + args.coe_rec \
        + '-te_key_dim' + str(args.te_key_dim)

    # + '-ss' + args.c_ss + '-sh' + args.c_sh + '-hs' + args.c_hs + '-hh' + args.c_hh \

    print('Model name:', auto_model_name)

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
    collate_fn = get_padding_fn(['s', 'a', 't', 'unified_t', 'k'])
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
    rec_config = RecNetConfig(
        n_embd=args.n_embd,
        n_head=args.n_head,
        attn_pdrop=float(args.dropout),
        resid_pdrop=float(args.dropout),
        embd_pdrop=float(args.dropout),
        block_size=args.context_length,
        n_layer=args.n_rec_layer,
        max_timestep=train_dataset.max_steps,
    )
    if args.sa_type == 'resfc':
        sa_config = ImplicitSAResFCConfig(
            n_embd=args.n_embd,
            block_size=args.context_length,
            n_state_layer=args.n_state_layer,
            n_action_layer=args.n_action_layer,
            max_timestep=train_dataset.max_steps,
            use_pos_emb=args.use_pos_emb
        )
    elif args.sa_type == 'gpt':
        sa_config = ImplicitSAGPTConfig(
            n_embd=args.n_embd,
            n_head=args.n_head,
            attn_pdrop=float(args.dropout),
            resid_pdrop=float(args.dropout),
            embd_pdrop=float(args.dropout),
            block_size=args.context_length,
            n_layer=args.n_state_layer+args.n_action_layer,
            state_layer=args.n_state_layer-1,
            max_timestep=train_dataset.max_steps
        )
    elif args.sa_type == 'egpt':
        sa_config = ExplicitSAGPTConfig(
            n_embd=args.n_embd,
            n_head=args.n_head,
            attn_pdrop=float(args.dropout),
            resid_pdrop=float(args.dropout),
            embd_pdrop=float(args.dropout),
            block_size=args.context_length,
            n_layer=args.n_action_layer,
            n_state_layer=args.n_state_layer,
            max_timestep=train_dataset.max_steps
        )
    elif args.sa_type == 'egpthn':
        sa_config = ExplicitSAHNGPTConfig(
            n_embd=args.n_embd,
            n_head=args.n_head,
            attn_pdrop=float(args.dropout),
            resid_pdrop=float(args.dropout),
            embd_pdrop=float(args.dropout),
            block_size=args.context_length,
            n_layer=args.n_action_layer,
            n_state_layer=args.n_state_layer,
            use_skip=args.use_skip,
            max_timestep=train_dataset.max_steps
        )
    elif args.sa_type == 'hn':
        sa_config = ExplicitSAHNConfig(
            dim_h=args.n_embd * args.n_state_layer,
            block_size=args.context_length,
            use_pos_emb=args.use_pos_emb,
            reward_layer=args.n_state_layer,
            max_timestep=train_dataset.max_steps
        )
    else:
        # should not reach here
        print('unknown sa_config.type')
        assert False

    optimizer_config = {
        'init_lr': float(args.init_lr),
        'weight_decay': float(args.weight_decay),
        'beta1': float(args.beta1),
        'beta2': float(args.beta2),
        'coe_cluster': float(args.coe_cluster),
        'coe_rec': float(args.coe_rec)
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
        sa_config=sa_config,
        rec_config=rec_config,
        vq_n_e=args.vq_n_e,
        KT=float(args.KT),
        optimizers_config=optimizer_config,
        scheduler_config=scheduler_config,
        state_dim=state_dim,
        action_dim=action_dim,
        key_dim=args.dim_key,
        e_dim=args.dim_e,
        te_keys_dim=None if (args.te_key_dim == 0) else args.te_key_dim
    )

    # autocot_model.configure_optimizers()
    model_path = os.path.join(MODEL_PATH, auto_model_name)
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

