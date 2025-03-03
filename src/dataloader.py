import os
import pandas as pd
import rasterio
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, default_collate

###############################################################################
#                  FONCTION COLLATE POUR IGNORER LES ÉCHANTILLONS None
###############################################################################
def skip_none_collate_fn(batch):
    """
    Filtre les éléments None avant de les assembler en batch.
    Si tous les éléments du batch sont None, renvoie None.
    """
    filtered_batch = [sample for sample in batch if sample is not None]
    if len(filtered_batch) == 0:
        return None  # Aucun exemple valide dans ce batch
    return default_collate(filtered_batch)

###############################################################################
#                         CLASSE FourBandSegDataset MODIFIÉE
###############################################################################
class FourBandSegDataset(Dataset):
    """
    Dataset pour la segmentation avec 4 canaux d'images (RGB+IR) et un masque
    correspondant. On lit les chemins dans un DataFrame qui comporte notamment :
      - 'planet_path'  : chemin .tif pour l'image Planet (4 canaux)
      - 'labels_path'  : chemin .tif pour le masque de segmentation
      - 'alignment'    : booléen (True/False)

    Les lignes avec alignment == True seulement sont utilisées.
    Si un fichier image ou label est introuvable ou "inconnu", on renvoie None 
    pour skipper cet échantillon.
    """
    def __init__(self, dataframe, transform=None):
        # 1) Ne garder que les lignes alignment == True
        df_filtered = dataframe[dataframe['alignment'] == True].copy().reset_index(drop=True)
        self.df = df_filtered
        self.transform = transform  # facultatif : pour data augmentation, etc.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        planet_tif = row["planet_path"]
        label_tif  = row["labels_path"]
        
        # Vérifier si les chemins sont valides
        if not planet_tif or planet_tif == "inconnu" or not os.path.exists(planet_tif):
            return None
        if not label_tif or label_tif == "inconnu" or not os.path.exists(label_tif):
            return None

        # Tentative de lecture du .tif 4 bandes
        try:
            with rasterio.open(planet_tif) as src:
                image = src.read()  # shape [4, H, W] si 4 canaux
        except:
            return None  # si la lecture échoue
        
        # Tentative de lecture du masque
        try:
            with rasterio.open(label_tif) as src:
                label = src.read(1)  # shape [H, W]
        except:
            return None  # skip si lecture masque échoue

        # Conversion en tenseurs PyTorch
        image_tensor = torch.from_numpy(image).float()  # [4, H, W]
        label_tensor = torch.from_numpy(label).long()   # [H, W]

        # (Optionnel) Appliquer des transformations/augmentations
        if self.transform is not None:
            pass
            # Exemple d'utilisation d'albumentations :
            # image_np = image_tensor.permute(1, 2, 0).numpy()  # (H, W, 4)
            # mask_np = label_tensor.numpy()                    # (H, W)
            #
            # transformed = self.transform(image=image_np, mask=mask_np)
            # new_img = transformed["image"]  # (H, W, 4)
            # new_msk = transformed["mask"]   # (H, W)
            #
            # image_tensor = torch.from_numpy(new_img).permute(2, 0, 1)
            # label_tensor = torch.from_numpy(new_msk)

        return image_tensor, label_tensor

