#!/bin/bash

cd ../src &&

# Example script for PickCube training (with a good set of hyper-parameters).
CUDA_VISIBLE_DEVICES=3 python train_cat.py \
   --model_name=TEST-1115-ACT-K0826 \
   --num_traj=500 --n_iters=3_600_000 --weight_decay=1e-3 --lr_schedule=cos_decay_with_warmup \
   --context_length=60 --model_type=s+a+k \
   --task=PegInsertionSide-v0 --key_state_coeff=0.1 \
   --n_layer=4 \
   --init_lr=5e-4 --num_workers=20 --save_every=2000 \
   --keys_name="keys8-26.txt"





# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --model_name=some_model_name \
#     --num_traj=500 --n_iters=1_600_000 \
#     --context_length=60 --model_type=s+a+cot \
#     --task=TurnFaucet-v0-v0 --key_state_coeff=0.1 \
#     --key_state_loss=0 --key_states=ab \
#     --init_lr=5e-4 --num_workers=20