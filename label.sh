#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=5 python label.py \
   --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=-1 \
   --model_name=TEST_PIS_GPTHN_NN_EMA_DGOAL_GGOAL_RECS_ENCO+_k4-r4-f1-c10_KT0.1_EMA0.95_temb1.2-r_l10.0-egpthn_s2_a1-emb128-key128-e128-cluster0.001-rec1.0 \
   --from_ckpt=100 \
