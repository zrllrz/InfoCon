#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=0 python label.py \
   --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=10 \
   --model_name=TEST_GPTHN_NN_EMA_EXPL_CLUS_NODISP_GOAL_RECS_NOTE-k4-r4-c10_KT0.1_EMA0.5_LIP1.0001-egpthn_s2_a1-emb128-key128-e128-cluster0.1-rec1.0-ts1.2 \
   --from_ckpt=4000 \

