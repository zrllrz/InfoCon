#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=0 python label.py \
   --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=10 \
   --model_name=GPTHN_NN_DISGOAL_RECS_WITH_KSREG-k4-r4-c10_KT1.0_LIP1.0001-egpthn_s2_a1-emb128-key128-e128-cluster0.00001-rec1.0-te_key_dim32 \
   --from_ckpt=2000 \

