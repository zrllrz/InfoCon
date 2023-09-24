#!/bin/bash

TASK=PegInsertionSide-v0

cd src &&

CUDA_VISIBLE_DEVICES=4 python label_clip.py --task=$TASK
