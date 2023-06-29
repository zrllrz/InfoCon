#!/bin/bash

cd src &&

# Example script for PickCube training (with a good set of hyper-parameters).
CUDA_VISIBLE_DEVICES=3 python train.py \
   --n_key_layer=3 --use_skip=False \
   --vq_len=30 --vq_beta=0.2 \
   --vq_kmeans_reset=1000 --vq_kmeans_step=100 \
   --key_states=a --n_act_layer=3 \
   --n_rec_layer=3 \
   --model_name=6_29_3-l_30-b_0.2-ks_a-kn_3-an_3-rn_3-prevention \
   --task=PegInsertionSide-v0 --seed=0 --num_traj=-1 \
   --multiplier=30 --num_workers=4
