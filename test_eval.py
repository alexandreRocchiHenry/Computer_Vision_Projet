
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchgeo.models import resnet50, ResNet50_Weights
from torchgeo.models import FarSeg
import sys
import os
from tqdm import tqdm  # Importer tqdm pour les barres de progression
from torch.amp import GradScaler

sys.path.append(os.path.abspath("src"))
from dataloader import FourBandSegDataset
from dataloader import skip_none_collate_fn
from dataloader import evaluate_model

###############################################################################
# Phase 4 : Évaluation
###############################################################################

farseg_best = FarSeg(backbone="resnet50", classes=8, backbone_pretrained=False)

# Modify the first convolutional layer for 4-channel input
old_conv = farseg_best.backbone.conv1
new_conv = nn.Conv2d(
    in_channels=4, out_channels=old_conv.out_channels, kernel_size=old_conv.kernel_size,
    stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None),
)

# Initialize the first three channels with trained weights and the fourth with random values
state_dict = torch.load("models/farseg_model.pth")
new_conv.weight.data[:, :3, :, :] = state_dict["backbone.conv1.weight"][:, :3, :, :]
nn.init.kaiming_normal_(new_conv.weight.data[:, 3:4, :, :])  # Random init for the 4th channel

# Replace the conv1 layer in the model
farseg_best.backbone.conv1 = new_conv

# Load the modified state dict (ignoring strict check)
farseg_best.load_state_dict(state_dict, strict=False)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using the device : ", device)
print("Nombre de GPUs :", torch.cuda.device_count())

farseg_best.to(device)
farseg_best.eval()

if torch.cuda.device_count() > 1:
    farseg = nn.DataParallel(farseg, device_ids=list(range(torch.cuda.device_count())))

farseg = farseg.to(device)
# 2) Évaluation sur le test set
test_loss, test_miou = evaluate_model(
    farseg_best,
    test_loader,     # DataLoader créé pour l’ensemble de test
    criterion,       # Même loss que pour l'entraînement
    device=device, 
    num_classes=8
)

print(f"Test Loss : {test_loss:.4f}")
print(f"Test mIoU : {test_miou:.4f}")