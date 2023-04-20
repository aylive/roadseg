from typing import Callable
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as _Dataset
from torch.utils.data import DataLoader as _DataLoader



class Trainer:

    def __init__(self, device, model):
        super().__init__()
        self.device = device
        self.model = model.to(device)

        self.criterion = NotImplementedError
        self.optimizer = NotImplementedError
        self.metrics = NotImplementedError

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
    def train(self, dataloader, epochs, criterion, optimizer):
        self.ave_epoch_loss = 0
        
        for i in range(epochs):
            self.criterion = criterion
            self.optimizer =  optimizer

            epoch_loss = self._run_on_(dataloader, is_training=True)

            self.ave_epoch_loss += epoch_loss
        
        self.ave_epoch_loss /= (i + 1)
        
    def evaluate(self, dataloader, criterion: Callable, metrics: Callable):
        self.criterion = criterion
        self.metrics = metrics
        self.eval_loss = self._run_on_(dataloader, is_training=False)