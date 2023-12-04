import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TASK = "PickCube-v0"
KEY = "keys-8-27-0"

key_path = "../data/" + TASK + "/" + KEY + ".txt"
print(key_path)

n_traj = 0
key_idx = np.array([0] * 10)
freq = np.array([0.0] * 10)

with open(key_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        n_traj += 1
        line = line.split(sep=',')[:-1]
        line = [int(item) for item in line]
        print(line)
        line_hot = np.array([int(item != -1) for item in line])
        print(line_hot)
        key_idx += line_hot

freq = key_idx / n_traj
print(freq)
