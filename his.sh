#!/bin/bash

cd src &&

python his.py \
  --task=PickCube-v0 \
  --control_mode=pd_joint_delta_pos \
  --obs_mode=state \
  --seed=0 \
  --key_name=key_test.txt