#!/bin/bash

cd ../src &&

# Example script for PickCube CoTPC training
CUDA_VISIBLE_DEVICES=0 python train.py \
   --model_name=PC_TEST \
   --num_traj=500 --n_iters=2_000_000 --weight_decay=1e-3 --lr_schedule=cos_decay_with_warmup \
   --context_length=60 --model_type=s+a+cot --batch_size=512 \
   --task=PickCube-v0 --key_state_coeff=0.1 \
   --n_layer=4 --vq_n_e=10 --key_state_loss=0 --key_states=ab \
   --init_lr=1e-3 --num_workers=20 --save_every=10 \
   --keys_name="key_test.txt"
