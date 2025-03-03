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

sys.path.append(os.path.abspath("src"))
from dataloader import FourBandSegDataset
from dataloader import skip_none_collate_fn

# Lecture du CSV
df_all = pd.read_csv("df_merged_extended.csv")

# Instancier le Dataset
train_dataset = FourBandSegDataset(df_all)

# Créer le DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=skip_none_collate_fn,
)
print("train_loader created!")

# Charger le ResNet pré-entraîné
model_resnet = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO)
state_dict_res = model_resnet.state_dict()
state_dict_res.pop("conv1.weight", None)
state_dict_res.pop("conv1.bias", None)

# Créer le FarSeg
farseg = FarSeg(backbone="resnet50", classes=8, backbone_pretrained=False)

# Modifier la première couche convolutive
old_conv = farseg.backbone.conv1
new_conv = nn.Conv2d(
    in_channels=4,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=(old_conv.bias is not None),
)
new_conv.weight.data[:, :3, :, :] = model_resnet.conv1.weight.data[:, :3, :, :]
nn.init.kaiming_normal_(new_conv.weight.data[:, 3:4, :, :])
farseg.backbone.conv1 = new_conv
farseg.backbone.load_state_dict(state_dict_res, strict=False)
print("Model Created")

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True

print("Start freezing all")
freeze_all_layers(farseg.backbone)
unfreeze_module(farseg.backbone.layer4)

for name, param in farseg.named_parameters():
    if "head" in name:
        param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using the device : ", device)
farseg = farseg.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, farseg.parameters()), lr=1e-4)

num_epochs_phase1 = 5
num_epochs_phase2 = 5
num_epochs_phase3 = 5
print("Start the training")

###############################################################################
# Phase 1 : Entraîner layer4 + head FarSeg
###############################################################################
for epoch in range(num_epochs_phase1):
    farseg.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{num_epochs_phase1} (Phase 1)")
    
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = farseg(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

###############################################################################
# Phase 2 : Débloquer layer3
###############################################################################
unfreeze_module(farseg.backbone.layer3)
for param_group in optimizer.param_groups:
    param_group["lr"] = 1e-5

for epoch in range(num_epochs_phase2):
    farseg.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{num_epochs_phase2} (Phase 2)")
    
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = farseg(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

###############################################################################
# Phase 3 : Débloquer tout le backbone
###############################################################################
unfreeze_module(farseg.backbone.layer2)
unfreeze_module(farseg.backbone.layer1)
unfreeze_module(farseg.backbone.conv1)

for param_group in optimizer.param_groups:
    param_group["lr"] = 5e-6

for epoch in range(num_epochs_phase3):
    farseg.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{num_epochs_phase3} (Phase 3)")
    
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = farseg(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

print("Entraînement terminé !")
