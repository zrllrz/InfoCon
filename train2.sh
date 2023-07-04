#!/bin/bash

cd src &&

# Example script for PickCube training (with a good set of hyper-parameters).
# --model_name=independent-sin_k-lay_3-10_cod-smo_None-sub \
CUDA_VISIBLE_DEVICES=7 python train2.py \
    --n_iters=3600000 --batch_size=256 \
    --init_lr=5e-4 --weight_decay=0.001 --lr_schedule=cos_decay_with_warmup --t_warmup=1000 \
    --beta1=0.9 --beta2=0.95 \
    --dropout=0.0 --n_head=8 --n_embd=128 --sub_pos=True \
    --n_key_layer=3 \
    --vq_n_e=10 --vq_beta=0.2 --vq_legacy=False --vq_smooth=None --vq_log=True --vq_kmeans_reset=1000 --vq_kmeans_step=100 \
    --n_act_layer=3 --seq_k=False \
    --commit=independent --n_commit_layer=3 \
    --model_name=test_legacy_true \
    --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state --seed=0 \
    --num_traj=-1 --context_length=60 --min_seq_length=60 \
    --save_every=1000 --log_every=1000 \
    --num_workers=4 --multiplier=26
