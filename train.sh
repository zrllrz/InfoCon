#!/bin/bash

cd src &&

# Example script for PickCube training (with a good set of hyper-parameters).
# --model_name=independent-sin_k-lay_3-10_cod-smo_None-sub \
CUDA_VISIBLE_DEVICES=1 python train.py \
    --n_iters=5_000_000 --batch_size=256 \
    --init_lr=5e-4 --weight_decay=0.001 --lr_schedule=cos_decay_with_warmup --t_warmup=1000 \
    --beta1=0.9 --beta2=0.95 \
    --dropout=0.0 --n_head=8 --n_embd=128 \
    --n_rec_layer=3 --n_key_layer=3 \
    --vq_n_e=10 --vq_beta=0.2 --vq_legacy=True \
    --vq_elastic=True --coe_loss_tolerance=1.0 --coe_rate_tolerance=0.5 \
    --vq_log=True --vq_persistence=none --vq_kmeans_reset=1000 --vq_kmeans_step=100\
    --n_act_layer=3 \
    --n_commit_layer=3 \
    --subgoal_marginal=1e-6 \
    --model_name=rkac-elastic_l_1.0_r_0.5-zerots-contrast-legacy-code_10-subgoal-detach-persistence_none_wutreset-keg_0.1 \
    --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state --seed=0 \
    --num_traj=-1 --context_length=60 --min_seq_length=60 \
    --save_every=1000 --log_every=1000 \
    --num_workers=6 --multiplier=26
