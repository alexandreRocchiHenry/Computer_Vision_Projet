import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from datetime import datetime, timedelta
from PIL import Image
import calendar
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

df_merged = pd.read_csv("df_merged.csv")

class CustomDataset():
    def __init__(self, df, start_date, end_date, transform=None):
        self.df = df
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.transform = transform

        # Créer une liste de dates à partir de la période spécifiée
        self.dates = self._generate_dates()

    def _generate_dates(self):
        dates = []
        current_date = self.start_date

        # Ajouter la première date (start_date) directement
        dates.append(current_date.strftime("%Y-%m-%d"))

        # Passer au mois suivant pour les dates suivantes
        current_date = current_date.replace(day=1)  # Passer au premier jour du mois suivant

        while current_date <= self.end_date:
            dates.append(current_date.strftime("%Y-%m-%d"))
            
            # Passer au mois suivant
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1, day=1)

        return dates

    def __len__(self):
        return len(self.dates) * len(self.df)  # Nombre d'images pour chaque date

    def __getitem__(self, idx):
        # Identifier la ligne du DataFrame et la date
        row_idx = idx // len(self.dates)  # Identifier la ligne du DataFrame
        date_idx = idx % len(self.dates)  # Identifier la date à partir de la liste des dates
        date_str = self.dates[date_idx]

        # Récupérer les chemins des images et des labels
        planet_path = self.df["planet_path"].iloc[row_idx]
        labels_path = self.df["labels_path"].iloc[row_idx]
        clean_label_path = labels_path.rstrip("/")
        label_filename  = clean_label_path.split("/")[-1]
        
        # Générer le chemin complet de l'image et des labels pour cette date
        # image_path = os.path.join(planet_path, f"{date_str}.tif")
        # label_path = os.path.join(labels_path, f"{label_filename}-{date_str.replace('-', '_')}.tif")

        # Générer les chemins avec les deux formats de date
        image_path_hyphen = os.path.join(planet_path, f"{date_str}.tif")
        image_path_underscore = os.path.join(planet_path, f"{date_str.replace('-', '_')}.tif")

        label_path_hyphen = os.path.join(labels_path, f"{label_filename}-{date_str}.tif")
        label_path_underscore = os.path.join(labels_path, f"{label_filename}-{date_str.replace('-', '_')}.tif")

        # Vérifier l'existence et choisir le bon chemin
        image_path = image_path_hyphen if os.path.exists(image_path_hyphen) else image_path_underscore
        label_path = label_path_hyphen if os.path.exists(label_path_hyphen) else label_path_underscore

        with rasterio.open(image_path) as src:
            image = src.read().astype("float32")  # [C, H, W]

        with rasterio.open(label_path) as src:
            label = src.read().astype("float32")  # [C, H, W]

        image = image[:3,:,:]

        image = np.transpose(image, (1, 2, 0))  # Changer de [C, H, W] à [H, W, C]
        label = np.transpose(label, (1, 2, 0))  # Changer de [C, H, W] à [H, W, C]

        # Appliquer les transformations après conversion en ndarray
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # Convertir en tensor après transformation
        image = torch.tensor(image).permute(2, 0, 1)  # [H, W, C] à [C, H, W]
        label = torch.tensor(label).permute(2, 0, 1)  # [H, W, C] à [C, H, W]

        return image, label

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=7):
        super(UNet, self).__init__()
        
        # Contracting Path (Encoder)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)
        
        # Expanding Path (Decoder)
        self.decoder5 = self.upconv_block(1024, 512)
        self.decoder4 = self.upconv_block(512 + 512, 256)  # Corrected to handle concatenation with 512 + 512
        self.decoder3 = self.upconv_block(256 + 256, 128)  # Adjusted for concatenation
        self.decoder2 = self.upconv_block(128 + 128, 64)   # Adjusted for concatenation
        self.decoder1 = self.upconv_block(128, out_channels)  # Adjusted for concatenation
        
        # Final output layer (Softmax activation for multiclass)
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Contracting path
        enc1 = self.encoder1(x)
        print("Enc1", enc1.shape)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        print("Enc2", enc2.shape)
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        print("Enc3", enc3.shape)
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        print("Enc4", enc4.shape)
        enc5 = self.encoder5(F.max_pool2d(enc4, 2))
        print("Enc5", enc5.shape)
        
        # Expanding path
        dec5 = self.decoder5(enc5)
        print("Dec5", dec5.shape)
        dec4 = self.decoder4(torch.cat([dec5, enc4], 1))
        print("Dec4", dec4.shape)
        dec3 = self.decoder3(torch.cat([dec4, enc3], 1))
        print("Dec3", dec3.shape)
        dec2 = self.decoder2(torch.cat([dec3, enc2], 1))
        print("Dec2", dec2.shape)
        dec1 = self.decoder1(torch.cat([dec2, enc1], 1))
        print("Dec1", dec1.shape)
        
        # Final output layer
        out = self.final_conv(dec1)
        
        return out

class ConvNeXtUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=7):
        super(ConvNeXtUNet, self).__init__()

        # Charger le modèle ConvNeXt pré-entraîné
        convnext = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)  # Utilisation du ConvNeXt Tiny pré-entraîné
        
        # Extraire les sorties intermédiaires de l'encodeur ConvNeXt
        self.encoder1 = convnext.features[0]  # Première couche de ConvNeXt
        self.encoder2 = convnext.features[1]  # Seconde couche de ConvNeXt
        self.encoder3 = convnext.features[2]  # Troisième couche de ConvNeXt
        self.encoder4 = convnext.features[3]  # Quatrième couche de ConvNeXt
        
        # Expanding Path (Decoder)
        self.decoder4 = self.upconv_block(768, 384)  # 768 → 384
        self.decoder3 = self.upconv_block(384 + 384, 192)  # (384 + skip 384) → 192
        self.decoder2 = self.upconv_block(192 + 192, 96)  # (192 + skip 192) → 96
        self.decoder1 = self.upconv_block(96 + 96, 64)  # (96 + skip 96) → 64


        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # Tête de sortie

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder : passage par les couches de ConvNeXt
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Expanding Path : Décodeur classique U-Net
        dec4 = self.decoder4(enc4)
        dec3 = self.decoder3(torch.cat([dec4, enc3], dim=1))  # Fusion des sorties encoder et decoder
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))

        # Final output layer
        out = self.final_conv(dec1)
        
        return out

def train(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            print("Optimizing")
            optimizer.zero_grad()
            print("Output")
            outputs = model(images)
            print("Loss")
            loss = criterion(outputs, labels)
            print("Backward")
            loss.backward()
            print("Step")
            optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}, Loss {loss.item()}")
    print("Training finished")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_date = "2018-05-01"
    end_date = "2019-12-31"

    print("\n")
    print("-------------------------------------")
    print("Creating CustomDataset and DataLoader")
    print("-------------------------------------")
    print("\n")

    dataset = CustomDataset(df_merged, start_date, end_date)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("\n")
    print("-------------------------------------")
    print("Creating UNet and ConvNeXtUNet models")
    print("-------------------------------------")
    print("\n")

    model_unet = UNet(in_channels=3, out_channels=7)
    model_unet = model_unet.to(device)
    # model_convunet = ConvNeXtUNet(in_channels=3, out_channels=7)

    criterion = nn.CrossEntropyLoss()
    optimizer_unet = torch.optim.Adam(model_unet.parameters(), lr=0.001)
    # optimizer_convunet = torch.optim.Adam(model_convunet.parameters(), lr=0.001)

    # print("Training UNet model")

    print("\n")
    print("-------------------------------------")
    print("Training ConvNext + UNet model")
    print("-------------------------------------")
    print("\n")

    train(model_unet, dataloader, criterion, optimizer_unet, num_epochs=1, device=device)

    # train(model_convunet, dataloader, criterion, optimizer_convunet, num_epochs=1)

    # convnext = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)  # Utilisation du ConvNeXt Tiny pré-entraîné

    # print(convnext.features)