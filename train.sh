#!/bin/bash

cd src &&

# Example script for PickCube InfoCon training:
CUDA_VISIBLE_DEVICES=0 python train.py \
    --n_iters=1_616_101 --batch_size=256 \
    --init_lr=1e-4 --weight_decay=0.001 --lr_schedule=cos_decay_with_warmup --t_warmup=1000 \
    --beta1=0.9 --beta2=0.95 --coe_cluster=0.001 --coe_rec=0.1 \
    --dropout=0.0 --n_head=8 --n_embd=128 --dim_key=128 --dim_e=128 \
    --n_key_layer=4 --n_rec_layer=4 --n_future_layer=2 \
    --vq_n_e=10 --vq_coe_ema=0.9 --KT=0.1 --vq_ema_ave --vq_use_st_emb --vq_st_emb_rate=1.2 --vq_coe_r_l1=0.0 --vq_use_r \
    --sa_type=egpthn --n_state_layer=2 --n_action_layer=1 \
    --model_name=PC_TEST_ \
    --task=PickCube-v0 --control_mode=pd_joint_delta_pos --obs_mode=state --seed=0 \
    --num_traj=-1 --context_length=60 --min_seq_length=60 \
    --save_every=10 --log_every=100 \
    --num_workers=6 --multiplier=26 \
    --train_mode=finetune
