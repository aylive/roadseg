import os

import wandb
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

from utils.datasets import SatImage
from models._dla import DLAWithResnet
from utils.utils import *
from test import _create_valid_data_loader

#
wandb.init(
    project="sat2road",
    name="230320_bench",
    mode="offline",
    config={
        "input_dir": '/home/aylive/workspace/peng_pku/data/20cities_tiles_256',
        "output_dir": '/home/aylive/workspace/peng_pku/outputs/tmp/230320/smp',
        "epochs": 1,
    }
)
    
cfg = wandb.config

# Set 
input_dir = cfg.input_dir   #'/home/aylive/workspace/peng_pku/data/deepglobe'

# Set
output_dir = cfg.output_dir   #'/home/aylive/workspace/peng_pku/outputs/{DATASET}/{DATE}'

# Set the device for computing
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

# Set training parameters
epochs = cfg.epochs


def _create_data_loader(meta_path, preprocessing_fn, batch_size, n_cpu):
    """
    Create a DataLoader for training.
    """
    class_rgb_values = np.array(
        [[0, 0, 0],   # background
         [255, 255, 255]]    # road
    )
    dataset = SatImage(
        meta_path,
        class_rgb_values=class_rgb_values,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )
    return dataloader


def run():

    # Create output directories if missing
    os.makedirs(output_dir, exist_ok=True)

    # ============
    # Create Model
    # ============
    encoder = 'resnet50'
    encoder_weights = 'imagenet'

    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=2,
        activation='sigmoid',
    )
    # model = DLAWithResnet(
    #     in_chans=3,
    #     out_chans=2,
    # )
    # model = model.to(device)

    # =================
    # Create Dataloader
    # =================
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    # Load training dataloader
    dataloader = _create_data_loader(
        meta_path=os.path.join(input_dir, 'metadata_train.csv'),
        preprocessing_fn=preprocessing_fn,
        batch_size=5,
        n_cpu=8,
    )

    # Load validation dataloader
    valid_dataloader = _create_valid_data_loader(
        meta_path=os.path.join(input_dir, 'metadata_valid.csv'),
        preprocessing_fn=preprocessing_fn,
        batch_size=5,
        n_cpu=8,
    )

    # ================================
    # Create loss, optimizer & metrics
    # ================================
    loss = smp_utils.losses.DiceLoss()

    optimizer = optim.Adam([
        dict(params=model.parameters(), lr=1e-4),
    ])

    metrics = [
        smp_utils.metrics.IoU(threshold=0.5),
    ]

    # ========
    # Training
    # ========

    train_epoch = smp_utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    best_iou_score = 0.0

    for i in range(1, epochs+1):

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)

        # ------------
        # Log progress
        # ------------

        # wandb loggging
        wandb.log({
            "train_dice_loss": train_logs['dice_loss'],
            "valid_dice_loss": valid_logs['dice_loss'],
            "train_IoU": train_logs['iou_score'],
            "valid_IoU": valid_logs['iou_score'],
        })
        
        # -------------
        # Save progress
        # -------------

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(
                model, os.path.join(output_dir, 'best_model.pth'),
            )
            print('Model saved!')

        #TODO: learning rate scheduler
    
    wandb.finish()

if __name__ == "__main__":
    run()