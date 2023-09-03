#!/bin/bash

cd src &&

# Example script for PickCube training (with a good set of hyper-parameters).
# --model_name=independent-sin_k-lay_3-10_cod-smo_None-sub \
CUDA_VISIBLE_DEVICES=3 python train.py \
    --n_iters=404_101 --batch_size=256 \
    --init_lr=1e-4 --weight_decay=0.001 --lr_schedule=cos_decay_with_warmup --t_warmup=1000 \
    --beta1=0.9 --beta2=0.95 --coe_cluster=0.001 --coe_rec=0.1 \
    --dropout=0.0 --n_head=8 --n_embd=128 --dim_key=128 --dim_e=128 \
    --n_key_layer=4 --n_rec_layer=4 --n_future_layer=2 \
    --vq_n_e=10 --vq_coe_ema=0.9 --KT=0.1 --vq_ema_ave --vq_use_st_emb --vq_st_emb_rate=1.2 --vq_coe_r_l1=0.0 --vq_use_r \
    --sa_type=egpthn --n_state_layer=2 --n_action_layer=1 --use_pos_emb \
    --model_name=SC_0904_0708 \
    --task=StackCube-v0 --control_mode=pd_joint_delta_pos --obs_mode=state --seed=0 \
    --num_traj=-1 --context_length=60 --min_seq_length=60 \
    --save_every=100 --log_every=100 \
    --num_workers=6 --multiplier=26 \
    --train_mode=finetune




# save: 100
# log: 100
# n_iters: 202_101




# 202_101


# --vq_use_clip_decrease_r
# --vq_kmeans_reset=none --vq_kmeans_step=none
# --vq_decay_energy=0.1
# --vq_coe_structure=0.1
# --vq_legacy_energy=0.2
# --use_key_energy
# --model_name=k4-c10_KT1.0-gpt_s3_a1-emb128-key128-e1024 \
# --c_ss=1.0 --c_sh=0.01 --c_hs=0.1 --c_hh=0.001 --repulse \
# --use_ts --rate_ts=10.0

# GPTHN_NN_EMA_GOAL_RECS_TIME++LONERESET_CLUS0.1_
# GPTHN_NN_EMA_GOAL_RECS_TIME++LONERESET_CLUSCOS_ENCO+_

