#!/bin/bash

cd src &&

# Example script for PickCube training (with a good set of hyper-parameters).
CUDA_VISIBLE_DEVICES=0 python train2.py \
    --n_iters=4800000 --batch_size=256 \
    --init_lr=5e-4 --weight_decay=0 --lr_schedule=cos_decay_with_warmup --t_warmup=1000 \
    --beta1=0.9 --beta2=0.95 \
    --dropout=0.0 \
    --n_head=8 --n_embd=128 \
    --n_key_layer=4 \
    --vq_n_e=100 --vq_beta=0.2 --vq_legacy=False --vq_log=True --vq_kmeans_reset=1000 --vq_kmeans_step=100 \
    --n_act_layer=4 \
    --commit=act --n_commit_layer=4 \
    --model_name=TEST \
    --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state --seed=0 \
    --num_traj=-1 --context_length=60 --min_seq_length=60 \
    --save_every=1000 --log_every=1000 \
    --num_workers=5, --multiplier=52