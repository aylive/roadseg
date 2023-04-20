import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import random, tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations as album
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

from utils.datasets import SatImage
from models._dla import DLAWithResnet
from utils.utils import *

# Set 
input_dir = '/home/aylive/workspace/peng_pku/data/20cities_tiles_256'

#
output_dir = '/home/aylive/workspace/peng_pku/outputs/tmp/230320'

# Set the device for computing
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")


def _create_valid_data_loader(meta_path, preprocessing_fn, batch_size, n_cpu):
    class_rgb_values = np.array(
        [[0, 0, 0],   # background
         [255, 255, 255]]    # road
    )
    dataset = SatImage(
        meta_path,
        class_rgb_values=class_rgb_values,
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
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

    # model = smp.DeepLabV3Plus(
    #     encoder_name=encoder,
    #     encoder_weights=encoder_weights,
    #     classes=2,
    #     activation='sigmoid',
    # )
    model = DLAWithResnet(
        in_chans=3,
        out_chans=2,
    )
    model = model.to(device)

    model = torch.load(
        os.path.join(output_dir, 'best_model.pth'), map_location=device,
    )

    # =================
    # Create Dataloader
    # =================
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    dataloader = _create_valid_data_loader(
        meta_path=os.path.join(input_dir, 'metadata_train.csv'),
        preprocessing_fn=preprocessing_fn,
        batch_size=1,
        n_cpu=8,
    )

    # ====================
    # Create loss, metrics
    # ====================
    loss = smp_utils.losses.DiceLoss()

    metrics = [
        smp_utils.metrics.IoU(threshold=0.5),
    ]
    
    # =======
    # Testing
    # =======

    test_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True
    )

    test_logs = test_epoch.run(dataloader)

    print("Evaluation on Test Data: ")
    print(f"Mean IoU Score: {test_logs['iou_score']:.4f}")
    print(f"Mean Dice Loss: {test_logs['dice_loss']:.4f}")


if __name__ == "__main__":
    run()