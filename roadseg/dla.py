import os
from tqdm import tqdm
from typing import Callable

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as _Dataset
from torch.utils.data import DataLoader as _DataLoader
import torch.nn.functional as F

from kornia.losses import DiceLoss

from models._dla import DLAWithResnet
import segmentation_models_pytorch as smp
from train import _create_data_loader
from trainer import Trainer

#TODO: 
# 1. loss tracking & logging (by wandb)
# 2. decorator for dataloader
# 3. other metrics tracking & logging (by wandb)

# #TODO: self-defined dataset for loading target_prob & target_vector
# class Dataset(_Dataset):
#     def __init__(self) -> None:
#         super().__init__()
    
#     def __getitem__(self, index):
#         return super().__getitem__(index)
    
#     def __len__(self):
#         pass

# #TODO: self-defined criterion combining L2-loss & CE-loss
# class Criterion(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
        
#     def forward(self, input, target):
#         pass

class Trainer:

    def __init__(self, device, model):
        super().__init__()
        self.device = device
        self.model = model.to(device)

        self.criterion = NotImplementedError
        self.optimizer = NotImplementedError

    def _run_on_(self, dataloader, is_training=True):
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        
        ave_loss = 0

        for idx, (x, y) in tqdm(enumerate(dataloader)):
            if is_training:
                self.optimizer.zero_grad()
        
            x, y = x.to(self.device), y.to(self.device)
            y = torch.argmax(y, dim=1)
            
            y_pred = self.model(x)

            loss = self.criterion(y_pred, y)

            if is_training:
                loss.backward()
                self.optimizer.step()

            # statistics
            ave_loss += loss.detach().item()
        
        ave_loss /= (idx + 1)
        
        return ave_loss 

    #TODO: criterion & optimizer
    def train(self, dataloader, epochs):
        self.ave_epoch_loss = 0
        
        for i in range(epochs):
            self.criterion = DiceLoss()
            self.optimizer =  optim.Adam([
                dict(params=model.parameters(), lr=1e-4),
            ])

            epoch_loss = self._run_on_(dataloader, is_training=True)

            self.ave_epoch_loss += epoch_loss
        
        self.ave_epoch_loss /= (i + 1)
        
    def evaluate(self, dataloader):
        self.eval_loss = self._run_on_(dataloader, is_training=False)

# For test only
if __name__ == "__main__":

    # device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    # model
    model = DLAWithResnet(
        in_chans=3,
        out_chans=2, #(2 + 4 * self.MaxDegree),
        base_chans=12,
        resnet_step=8,
    )
    
    # trainer
    pandora = Trainer(device, model=model)

    # data
    input_dir = '/home/aylive/workspace/peng_pku/data/20cities_tiles_256'

    encoder = 'resnet50'
    encoder_weights = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    dataloader = _create_data_loader(
        meta_path=os.path.join(input_dir, 'metadata_train.csv'),
        preprocessing_fn=preprocessing_fn,
        batch_size=5,
        n_cpu=8,
    )

    #
    criterion = DiceLoss()
    optimizer = optim.Adam([
        dict(params=model.parameters(), lr=1e-4),
    ])
    pandora.train(dataloader=dataloader, epochs=1)