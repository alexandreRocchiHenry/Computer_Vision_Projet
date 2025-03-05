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
from dataloader import AugmentedDataset
from dataloader import skip_none_collate_fn
from dataloader import evaluate_model

# 1. Lecture du CSV complet
df_all = pd.read_csv("df_merged_extended.csv")

# 2. Filtrer uniquement les lignes alignées
df_filtered = df_all[df_all["alignment"] == True].copy().reset_index(drop=True)

# 3. Mélanger les données et definir les df pour chaque continent
df_filtered_shuffled = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)

df_afrique = df_filtered_shuffled[df_filtered_shuffled["continent"] == "Africa"].copy().reset_index(drop=True)
df_asie = df_filtered_shuffled[df_filtered_shuffled["continent"] == "Asia"].copy().reset_index(drop=True)
df_europe = df_filtered_shuffled[df_filtered_shuffled["continent"] == "Europe"].copy().reset_index(drop=True)
df_north_america = df_filtered_shuffled[df_filtered_shuffled["continent"] == "North_america"].copy().reset_index(drop=True)
df_oceania = df_filtered_shuffled[df_filtered_shuffled["continent"] == "Oceania"].copy().reset_index(drop=True)
df_south_america = df_filtered_shuffled[df_filtered_shuffled["continent"] == "South_america"].copy().reset_index(drop=True)

df_without_afrique = df_filtered_shuffled[~df_filtered_shuffled["continent"].isin(["Africa"])].copy().reset_index(drop=True)
df_without_asie = df_filtered_shuffled[~df_filtered_shuffled["continent"].isin(["Asia"])].copy().reset_index(drop=True)
df_without_europe = df_filtered_shuffled[~df_filtered_shuffled["continent"].isin(["Europe"])].copy().reset_index(drop=True)
df_without_north_america = df_filtered_shuffled[~df_filtered_shuffled["continent"].isin(["North_america"])].copy().reset_index(drop=True)
df_without_oceania = df_filtered_shuffled[~df_filtered_shuffled["continent"].isin(["Oceania"])].copy().reset_index(drop=True)
df_without_south_america = df_filtered_shuffled[~df_filtered_shuffled["continent"].isin(["South_america"])].copy().reset_index(drop=True)

# 4. Définir les proportions pour train / val / test
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Vérification: doit être égal à 1.0
assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "La somme des ratios doit faire 1"

n_total = len(df_filtered_shuffled)
n_train = int(train_ratio * n_total)
n_val = int(val_ratio * n_total)
n_test = n_total - (n_train + n_val)

# Ajout d'une partie pour les tests sur chaque continent et sans ce continent
# Europe et Monde sans Europe pris en exemple
n_total_wo_continent = len(df_without_europe)
n_total_continent = len(df_europe)
n_train_continent = int(train_ratio * n_total_wo_continent)
n_val_continent = int(val_ratio * n_total_wo_continent)
n_test_continent = int(test_ratio * n_total_continent)

# 5. Découpage en trois sous-ensembles
train_df = df_without_europe.iloc[:n_train].reset_index(drop=True)
val_df = df_without_europe.iloc[n_train:n_train+n_val].reset_index(drop=True)
test_df = df_europe.copy().reset_index(drop=True)

# 6. Instanciation des Dataset
train_dataset = FourBandSegDataset(train_df)
val_dataset = FourBandSegDataset(val_df)
test_dataset = FourBandSegDataset(test_df)

# 7. Création des DataLoader
train_loader = DataLoader(
    train_dataset, batch_size=8,
    shuffle=True, num_workers=4, collate_fn=skip_none_collate_fn,
)
val_loader = DataLoader(
    val_dataset, batch_size=8,
    shuffle=False, num_workers=4, collate_fn=skip_none_collate_fn,
)
test_loader = DataLoader(
    test_dataset, batch_size=8,
    shuffle=False, num_workers=4, collate_fn=skip_none_collate_fn,
)

print("Taille Entraînement :", len(train_dataset))
print("Taille Validation   :", len(val_dataset))
print("Taille Test         :", len(test_dataset))

print("Chargement DataLoaders terminé.")

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
    in_channels=4, out_channels=old_conv.out_channels, kernel_size=old_conv.kernel_size,
    stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None),
)
new_conv.weight.data[:, :3, :, :] = model_resnet.conv1.weight.data[:, :3, :, :]
nn.init.kaiming_normal_(new_conv.weight.data[:, 3:4, :, :])
farseg.backbone.conv1 = new_conv
farseg.backbone.load_state_dict(state_dict_res, strict=False)
print("Model Created")

# Fonctions pour gérer DataParallel
def get_model(farseg):
    """Retourne le modèle encapsulé dans DataParallel si nécessaire."""
    return farseg.module if isinstance(farseg, nn.DataParallel) else farseg

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True

print("Start freezing all")
freeze_all_layers(get_model(farseg).backbone)
unfreeze_module(get_model(farseg).backbone.layer4)

for name, param in get_model(farseg).named_parameters():
    if "head" in name:
        param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using the device : ", device)
print("Nombre de GPUs :", torch.cuda.device_count())

if torch.cuda.device_count() > 1:
    farseg = nn.DataParallel(farseg, device_ids=list(range(torch.cuda.device_count())))

farseg = farseg.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, farseg.parameters()), lr=1e-4)

num_epochs_phase1 = 30  
num_epochs_phase2 = 30
num_epochs_phase3 = 50
patience = 5 


print("Start the training")
###############################################################################
# Fonction de train
###############################################################################
scaler = GradScaler(enabled=True)  # Activation de l'AMP (désactivez en cas de problème)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

def train_phase(num_epochs, phase_name):
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        farseg.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{num_epochs} ({phase_name})")

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = farseg(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        val_loss, val_miou = evaluate_model(farseg, val_loader, criterion, device=device, num_classes=8)
        scheduler.step(val_loss)

        print(f"Validation Loss: {val_loss:.4f} | Validation mIoU: {val_miou:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(farseg.state_dict(), f"models/best_model_{phase_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

###############################################################################
# Training
###############################################################################

print("Starting Phase 1")

freeze_all_layers(get_model(farseg).backbone)
unfreeze_module(get_model(farseg).backbone.layer4)
train_phase(num_epochs_phase1, "Phase1")

print("Starting Phase 2")
unfreeze_module(get_model(farseg).backbone.layer3)
for param_group in optimizer.param_groups:
    param_group["lr"] = 1e-5
train_phase(num_epochs_phase2, "Phase2")

print("Starting Phase 3")
unfreeze_module(get_model(farseg).backbone.layer2)
unfreeze_module(get_model(farseg).backbone.layer1)
unfreeze_module(get_model(farseg).backbone.conv1)
for param_group in optimizer.param_groups:
    param_group["lr"] = 5e-6
train_phase(num_epochs_phase3, "Phase3")
print("Entraînement terminé !")


# save model
torch.save(get_model(farseg).state_dict(), "models/farseg_model.pth")

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