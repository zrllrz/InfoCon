import os
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict

from mani_skill2.utils.io_utils import load_json
import mani_skill2.envs
import torch

from autocot import (
    KeyNetConfig,
    ActNetConfig,
    AutoCoT,
)

from path import MODEL_PATH, DATA_PATH

@torch.no_grad
def predict(model, action_hist, state_hist, t):
    assert model.model_type in ['s', 's+a']
