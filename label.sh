#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=0 python label.py \
   --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=10 \
   --model_name=kanw-energy-legacy_c_0.2_e_0.2-struct_0.1-code_10 \
   --from_ckpt=1000 \

