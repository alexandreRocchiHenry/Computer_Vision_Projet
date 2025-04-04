import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchgeo.models import swin_v2_b, Swin_V2_B_Weights
from torchgeo.models import FarSeg
import sys
import os
from tqdm import tqdm  
from torch.amp import GradScaler

sys.path.append(os.path.abspath("src"))
from dataloader import FourBandSegDataset
from dataloader import skip_none_collate_fn
from dataloader import evaluate_model

df_all = pd.read_csv("df_merged_extended.csv")

df_filtered = df_all[df_all["alignment"] == True].copy().reset_index(drop=True)

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

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "La somme des ratios doit faire 1"

n_total = len(df_filtered_shuffled)
n_train = int(train_ratio * n_total)
n_val = int(val_ratio * n_total)
n_test = n_total - (n_train + n_val)

n_total_wo_continent = len(df_without_europe)
n_total_continent = len(df_europe)
n_train_continent = int(train_ratio * n_total_wo_continent)
n_val_continent = int(val_ratio * n_total_wo_continent)
n_test_continent = int(test_ratio * n_total_continent)

train_df = df_without_europe.iloc[:n_train].reset_index(drop=True)
val_df = df_without_europe.iloc[n_train:n_train+n_val].reset_index(drop=True)
test_df = df_europe.copy().reset_index(drop=True)

train_dataset = FourBandSegDataset(train_df)
val_dataset = FourBandSegDataset(val_df)
test_dataset = FourBandSegDataset(test_df)

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
farseg = FarSeg(backbone="resnet50", classes=8, backbone_pretrained=False)
old_conv = farseg.backbone.conv1
new_conv = nn.Conv2d(
    in_channels=4, out_channels=old_conv.out_channels, kernel_size=old_conv.kernel_size,
    stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None),
)
new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
nn.init.kaiming_normal_(new_conv.weight.data[:, 3:4, :, :])  # Initialisation pour le 4e canal

farseg.backbone.conv1 = new_conv

print("Model Created")

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

farseg.apply(initialize_weights)

def get_model(farseg):
    """Retourne le modèle encapsulé dans DataParallel si nécessaire."""
    return farseg.module if isinstance(farseg, nn.DataParallel) else farseg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using the device : ", device)
print("Nombre de GPUs :", torch.cuda.device_count())

if torch.cuda.device_count() > 1:
    farseg = nn.DataParallel(farseg, device_ids=list(range(torch.cuda.device_count())))

farseg = farseg.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(farseg.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
scaler = GradScaler(enabled=True)

scaler = GradScaler(enabled=True)  
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

def train_phase(num_epochs, phase_name):
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    for epoch in range(num_epochs):
        farseg.train()
        running_loss = 0.0
        print(f"📌 Epoch {epoch+1}/{num_epochs} ({phase_name})")

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

        print(f"✅ Validation Loss: {val_loss:.4f} | Validation mIoU: {val_miou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(farseg.state_dict(), f"models/best_model_{phase_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

print("🚀 Starting Full Training")
train_phase(num_epochs=50, phase_name="FullTraining")
torch.save(farseg.state_dict(), "models/farseg_final_v4.pth")

farseg.load_state_dict(torch.load("models/farseg_final_v4.pth"))
farseg.to(device)
farseg.eval()

test_loss, test_miou = evaluate_model(farseg, test_loader, criterion, device=device, num_classes=8)

print(f"🎯 Test Loss : {test_loss:.4f}")
print(f"🎯 Test mIoU : {test_miou:.4f}")
