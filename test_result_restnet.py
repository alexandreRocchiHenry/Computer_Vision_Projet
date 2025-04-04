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
df_north_america = df_filtered_shuffled[df_filtered_shuffled["continent"] == "North America"].copy().reset_index(drop=True)
df_oceania = df_filtered_shuffled[df_filtered_shuffled["continent"] == "Oceania"].copy().reset_index(drop=True)
df_south_america = df_filtered_shuffled[df_filtered_shuffled["continent"] == "South America"].copy().reset_index(drop=True)

df_without_afrique = df_filtered_shuffled[~df_filtered_shuffled["continent"].isin(["Africa"])].copy().reset_index(drop=True)
df_without_asie = df_filtered_shuffled[~df_filtered_shuffled["continent"].isin(["Asia"])].copy().reset_index(drop=True)
df_without_europe = df_filtered_shuffled[~df_filtered_shuffled["continent"].isin(["Europe"])].copy().reset_index(drop=True)
df_without_north_america = df_filtered_shuffled[~df_filtered_shuffled["continent"].isin(["North America"])].copy().reset_index(drop=True)
df_without_oceania = df_filtered_shuffled[~df_filtered_shuffled["continent"].isin(["Oceania"])].copy().reset_index(drop=True)
df_without_south_america = df_filtered_shuffled[~df_filtered_shuffled["continent"].isin(["South America"])].copy().reset_index(drop=True)
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
train_df = df_without_afrique.iloc[:n_train].reset_index(drop=True)
val_df = df_without_afrique.iloc[n_train:n_train+n_val].reset_index(drop=True)
test_df = df_north_america.copy().reset_index(drop=True)
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
data_iter = iter(test_loader)
images, labels = next(data_iter)  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images = images.to(device)
labels = labels.to(device)

def visualize_results(images, labels, predictions, num_samples=4):
    """
    Visualize original images, ground truth masks, and predicted segmentation masks.
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))
    
    for i in range(num_samples):
        img = images[i].cpu().numpy().transpose(1, 2, 0)  
        label = labels[i].cpu().numpy()
        pred = predictions[i].cpu().numpy()

        
        img_rgb = img[:, :, :3]  

        img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())
        img_rgb = (img_rgb * 255).astype("uint8")

        axes[i, 0].imshow(img_rgb) 


        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(label, cmap="jet")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="jet")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()
    

farseg_best = FarSeg(backbone="resnet50", classes=8, backbone_pretrained=False)
old_conv = farseg_best.backbone.conv1
new_conv = nn.Conv2d(
    in_channels=4, out_channels=old_conv.out_channels, kernel_size=old_conv.kernel_size,
    stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None),
)
state_dict = torch.load("models/farseg_model_north_america.pth")
new_conv.weight.data[:, :3, :, :] = state_dict["backbone.conv1.weight"][:, :3, :, :]
nn.init.kaiming_normal_(new_conv.weight.data[:, 3:4, :, :])  # Random init for the 4th channel
farseg_best.backbone.conv1 = new_conv
farseg_best.load_state_dict(state_dict, strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using the device : ", device)
print("Nombre de GPUs :", torch.cuda.device_count())

farseg_best.to(device)
farseg_best.eval()

if torch.cuda.device_count() > 1:
    farseg_best = nn.DataParallel(farseg_best, device_ids=list(range(torch.cuda.device_count())))

test_loss, test_miou, test_acc = evaluate_model(
    farseg_best,
    test_loader,     
    criterion,      
    device=device, 
    num_classes=8
)

print(f"Test Loss  : {test_loss:.4f}")
print(f"Test mIoU  : {test_miou:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")  

