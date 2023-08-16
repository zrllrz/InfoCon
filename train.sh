#!/bin/bash

cd src &&

# Example script for PickCube training (with a good set of hyper-parameters).
# --model_name=independent-sin_k-lay_3-10_cod-smo_None-sub \
CUDA_VISIBLE_DEVICES=5 python train.py \
    --n_iters=2_000_000 --batch_size=256 \
    --init_lr=5e-4 --weight_decay=0.001 --lr_schedule=cos_decay_with_warmup --t_warmup=1000 \
    --beta1=0.9 --beta2=0.95 --coe_cluster=0.5 --coe_rec=1.0 \
    --dropout=0.0 --n_head=8 --n_embd=128 --dim_key=128 --dim_e=128 \
    --n_key_layer=4 --n_rec_layer=4 \
    --vq_n_e=10 --vq_coe_ema=0.5 --KT=0.1 --vq_t_emb_rate=1.2 \
    --sa_type=egpthn --n_state_layer=2 --n_action_layer=1 --use_pos_emb \
    --model_name=TEST_GPTHN_NN_EMA_GOAL_RECS_TIME++- \
    --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state --seed=0 \
    --num_traj=-1 --context_length=60 --min_seq_length=60 \
    --save_every=1000 --log_every=1000 \
    --num_workers=6 --multiplier=26
# --vq_kmeans_reset=none --vq_kmeans_step=none
# --vq_decay_energy=0.1
# --vq_coe_structure=0.1
# --vq_legacy_energy=0.2
# --use_key_energy
# --model_name=k4-c10_KT1.0-gpt_s3_a1-emb128-key128-e1024 \
# --c_ss=1.0 --c_sh=0.01 --c_hs=0.1 --c_hh=0.001 --repulse \
# --use_ts --rate_ts=10.0
