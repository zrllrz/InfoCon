import os
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import MS2Demos, get_padding_fn
from autocot import (
    KeyNetConfig,
    FutureNetConfig,
    ExplicitSAHNGPTConfig,
    RecNetConfig,
    AutoCoT
)
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

    parser.add_argument("--n_rec_layer", default=4, type=int,
                        help="Number of attention layers in RecNet")

    parser.add_argument("--n_future_layer", default=4, type=int,
                        help="Number of attention layers in FutureNet")

    parser.add_argument('--vq_n_e', type=int, default=100,
                        help="How many kinds of keys in the key_book")
    parser.add_argument("--vq_use_r", action='store_true',
                        help="use learnable radius of prototype")
    parser.add_argument('--vq_coe_ema', type=str, default='0.95',
                        help="ema moving rate")
    parser.add_argument('--vq_ema_ave', action='store_true',
                        help="average or not")
    parser.add_argument('--KT', type=str, default='1.0',
                        help="Temperature for classifier")
    parser.add_argument("--vq_use_ft_emb", action='store_true',
                        help="use frequent time step embedding")
    parser.add_argument("--vq_use_st_emb", action='store_true',
                        help="use spherical time step embedding")
    parser.add_argument("--vq_st_emb_rate", default='1.2', type=str,
                        help="division rate for time sphere embedding")
    parser.add_argument("--vq_coe_r_l1", default='0.0', type=str,
                        help="l1 regularization on length of every prototype")
    parser.add_argument("--vq_use_prob_sel_train", action='store_true',
                        help="If true, using prob sample when training")
    parser.add_argument("--vq_use_timestep_appeal", action='store_true',
                        help="If true, prototype will move close to time in time interval")

    parser.add_argument("--coe_cluster", default='0.1', type=str, help="cluster weight")
    parser.add_argument("--coe_rec", default='1.0', type=str, help="reconstruction weight from key_soft")
    parser.add_argument("--use_decay_mask_rate", action='store_true',
                        help="mask cluster item when it's policy is too large")

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
    parser.add_argument("--use_future_state", action='store_true',
                        help="if True, we will append the future states back to ")

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
    parser.add_argument('--train_half', action='store_true',
                        help="train half (do not optimize gen goal loss)")
    parser.add_argument('--train_mode', default='scratch', type=str,
                        help="training mode")  # 3 choices: 'scratch', 'pretrain', 'finetune'

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    auto_model_name = \
        args.model_name \
        + 'k' + str(args.n_key_layer) \
        + '-r' + str(args.n_rec_layer) \
        + (('-f' + str(args.n_future_layer)) if args.n_future_layer != 0 else '') \
        + '-c' + str(args.vq_n_e) + ('-use_prob_sel_train' if args.vq_use_prob_sel_train else '') + ('-use_timestep_appeal' if args.vq_use_timestep_appeal else '')\
        + '_KT' + args.KT + '_EMA' + args.vq_coe_ema + ('_ema_ave' if args.vq_ema_ave else '_ema_single') + (('_st-emb' + args.vq_st_emb_rate) if args.vq_use_st_emb else '') \
        + ('_ft-emb' if args.vq_use_ft_emb else '') \
        + '-r_l1' + args.vq_coe_r_l1 + ('-use_r' if args.vq_use_r else '') \
        + '-' + args.sa_type + '_s' + str(args.n_state_layer) + '_a' + str(args.n_action_layer) + ('-use_future_state' if args.use_future_state else '') \
        + '-emb' + str(args.n_embd) \
        + '-key' + str(args.dim_key) \
        + '-e' + str(args.dim_e) \
        + '-cluster' + args.coe_cluster + '-rec' + args.coe_rec + ('-use_decay_mask_rate' if args.use_decay_mask_rate else '') \
        + ('-train_half' if args.train_half else '') \
        + ('-finetune' if args.train_mode == 'finetune' else '')

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
    print('Training epoch:', args.n_iters / (len(train_dataset) / args.batch_size))
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

    if args.n_future_layer != 0:
        future_config = FutureNetConfig(
            n_embd=args.n_embd,
            n_head=args.n_head,
            attn_pdrop=float(args.dropout),
            resid_pdrop=float(args.dropout),
            embd_pdrop=float(args.dropout),
            block_size=args.context_length,
            n_layer=args.n_future_layer,
            max_timestep=train_dataset.max_steps,
        )
    else:
        future_config = None

    if args.sa_type == 'egpthn':
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
            max_timestep=train_dataset.max_steps,
            use_future_state=args.use_future_state
        )
    else:
        # should not reach here
        print('suggest using sa_config.type = \'egpthn\'')
        assert False

    optimizer_config = {
        'init_lr': float(args.init_lr),
        'weight_decay': float(args.weight_decay),
        'beta1': float(args.beta1),
        'beta2': float(args.beta2),
        'coe_cluster': float(args.coe_cluster),
        'coe_rec': float(args.coe_rec),
        'use_decay_mask_rate': args.use_decay_mask_rate
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
        future_config=future_config,
        vq_n_e=args.vq_n_e,
        vq_use_r=args.vq_use_r,
        vq_coe_ema=float(args.vq_coe_ema),
        vq_ema_ave=args.vq_ema_ave,
        KT=float(args.KT),
        optimizers_config=optimizer_config,
        scheduler_config=scheduler_config,
        state_dim=state_dim,
        action_dim=action_dim,
        key_dim=args.dim_key,
        e_dim=args.dim_e,
        vq_use_ft_emb=args.vq_use_ft_emb,
        vq_use_st_emb=args.vq_use_st_emb,
        vq_st_emb_rate=float(args.vq_st_emb_rate),
        vq_coe_r_l1=float(args.vq_coe_r_l1),
        vq_use_prob_sel_train=args.vq_use_prob_sel_train,
        vq_use_timestep_appeal=args.vq_use_timestep_appeal,
        task=args.task
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

    if args.train_mode == 'scratch':
        autocot_model.mode = 'goal'
        mysavelogger = MySaveLogger(
            path=model_path,
            args=args,
            epoch_frequency=args.save_every
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            callbacks=[mysavelogger],
            max_steps=(args.n_iters // 2) if args.train_half else args.n_iters,
        )
        trainer.fit(
            model=autocot_model,
            train_dataloaders=train_data,
        )

    elif args.train_mode == 'pretrain':
        autocot_model.mode = 'rec'
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_steps=101*100,
        )
        trainer.fit(
            model=autocot_model,
            train_dataloaders=train_data,
        )
        print('Save pretrained checkpoints')
        torch.save({
            'module': autocot_model.state_dict(),
            'metadata': vars(args)
        }, os.path.join(MODEL_PATH, args.task + '_REC_CHECKPOINT' + ('_R' if args.vq_use_r else '') + '.pth'))
        print('Done!')

    elif args.train_mode == 'finetune':
        autocot_model.mode = 'goal'
        rec_path = os.path.join(MODEL_PATH, args.task + '_REC_CHECKPOINT' + ('_R' if args.vq_use_r else '') + '.pth')
        print('Try to load pretrained model from path:\n', rec_path)
        ckpt = torch.load(rec_path, map_location=torch.device('cpu'))
        state_dict_from_ckpt = ckpt['module']
        autocot_model.load_state_dict(state_dict_from_ckpt, strict=False)
        print('Done!')
        mysavelogger = MySaveLogger(
            path=model_path,
            args=args,
            epoch_frequency=args.save_every
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            callbacks=[mysavelogger],
            max_steps=(args.n_iters // 2) if args.train_half else args.n_iters,
        )
        trainer.fit(
            model=autocot_model,
            train_dataloaders=train_data,
        )

    else:
        print('Unknown train mode')
        assert False
