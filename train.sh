#!/bin/bash

cd src &&

# Example script for PickCube training (with a good set of hyper-parameters).
# --model_name=independent-sin_k-lay_3-10_cod-smo_None-sub \
CUDA_VISIBLE_DEVICES=4 python train.py \
    --n_iters=5_000_000 --batch_size=256 \
    --init_lr=5e-4 --weight_decay=0.001 --lr_schedule=cos_decay_with_warmup --t_warmup=1000 \
    --beta1=0.9 --beta2=0.95 \
    --dropout=0.0 --n_head=8 --n_embd=128 \
    --n_key_layer=4 \
    --vq_n_e=10 --vq_legacy_cluster=0.2 \
    --n_act_layer=2 \
    --n_e_layer=1 \
    --model_name=kanw-compliance-energy-legacy_c_0.2_e_0.2-struct_0.1-code_10 \
    --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state --seed=0 \
    --num_traj=-1 --context_length=60 --min_seq_length=60 \
    --save_every=1000 --log_every=1000 \
    --num_workers=6 --multiplier=26

# --vq_kmeans_reset=none --vq_kmeans_step=none
# --vq_decay_energy=0.1
# --vq_coe_structure=0.1
# --vq_legacy_energy=0.2
# --use_key_energy
