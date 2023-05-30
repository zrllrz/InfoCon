import os
import torch
from pytorch_lightning.callbacks import Callback


class MySaveLogger(Callback):
    def __init__(self, path, args, epoch_frequency=200):
        super().__init__()
        self.path = path
        self.epoch_freq = epoch_frequency
        self.args = args
        self.cnt = 0

    def log_model(self, pl_module):
        save_path = os.path.join(self.path, f'epoch{self.cnt}.pth')
        torch.save({
            'module': pl_module.state_dict(),
            'metadata': vars(self.args)
        }, save_path)

    def on_train_epoch_end(self, trainer, pl_module):
        self.cnt += 1
        if self.cnt % self.epoch_freq == 0:
            print("\nSave model before epoch: %d" % self.cnt)
            self.log_model(pl_module)
