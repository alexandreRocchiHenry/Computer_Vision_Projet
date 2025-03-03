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

sys.path.append(os.path.abspath("src"))
from dataloader import FourBandSegDataset

# Lecture du CSV (vous pouvez adapter le chemin vers votre fichier)
df_all = pd.read_csv("df_merged_extended.csv")

# Instancier le Dataset (seules les lignes alignment == True seront gardées)
train_dataset = FourBandSegDataset(df_all)

# Créer le DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,  # ajuster selon votre RAM
    shuffle=True,
    num_workers=4  # ou 0 si souci sur Windows
)
################################################################################
#   3) Charger un ResNet pré-entraîné, l'adapter à 4 canaux et insérer dans FarSeg
################################################################################
# Charger le ResNet pré-entraîné
model_resnet = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO)

# Extraire le state_dict
state_dict_res = model_resnet.state_dict()

# Supprimer les clés relatives à conv1
state_dict_res.pop("conv1.weight", None)
state_dict_res.pop("conv1.bias", None)

# Créer le FarSeg (par défaut: ResNet50 3 canaux)
farseg = FarSeg(backbone="resnet50", classes=8, backbone_pretrained=False)

# Remplacer la conv1 de FarSeg pour qu'elle accepte 4 canaux
old_conv = farseg.backbone.conv1
new_conv = nn.Conv2d(
    in_channels=4,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=(old_conv.bias is not None),
)

# Vous pouvez copier manuellement les 3 premiers canaux depuis model_resnet.conv1 si vous le souhaitez :
new_conv.weight.data[:, :3, :, :] = model_resnet.conv1.weight.data[:, :3, :, :]
nn.init.kaiming_normal_(new_conv.weight.data[:, 3:4, :, :])

farseg.backbone.conv1 = new_conv

# Charger le reste du state_dict (sans conv1)
farseg.backbone.load_state_dict(state_dict_res, strict=False)

################################################################################
#   4) Exemple de dégel progressif des couches (fine-tuning progressif)        #
################################################################################

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True

# Geler le backbone, sauf layer4 + head FarSeg
freeze_all_layers(farseg.backbone)
unfreeze_module(farseg.backbone.layer4)

# FarSeg a aussi des couches de "head", "fuse", etc. On les dégrèle toutes :
for name, param in farseg.named_parameters():
    if "head" in name:
        param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
farseg = farseg.to(device)

criterion = nn.CrossEntropyLoss()  # pour segmentation 8 classes
optimizer = optim.Adam(filter(lambda p: p.requires_grad, farseg.parameters()), lr=1e-4)

# Nombre d'époques pour chaque phase de dégel
num_epochs_phase1 = 5
num_epochs_phase2 = 5
num_epochs_phase3 = 5

###############################################################################
# Phase 1 : Entraîner layer4 + head FarSeg
###############################################################################
for epoch in range(num_epochs_phase1):
    farseg.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = farseg(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # Optionnel: validation, métriques, logs...

###############################################################################
# Phase 2 : Débloquer layer3
###############################################################################
unfreeze_module(farseg.backbone.layer3)

# Réduire le LR pour stabiliser l'entraînement
for param_group in optimizer.param_groups:
    param_group["lr"] = 1e-5

for epoch in range(num_epochs_phase2):
    farseg.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = farseg(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

###############################################################################
# Phase 3 : Débloquer tout le backbone (layer2, layer1, conv1)
###############################################################################
unfreeze_module(farseg.backbone.layer2)
unfreeze_module(farseg.backbone.layer1)
unfreeze_module(farseg.backbone.conv1)

for param_group in optimizer.param_groups:
    param_group["lr"] = 5e-6

for epoch in range(num_epochs_phase3):
    farseg.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = farseg(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 6) Optionnel : Entraîner plus longtemps le modèle totalement "dégelé"
#
# for epoch in range(num_epochs_extra):
#     ...

print("Entraînement terminé !")