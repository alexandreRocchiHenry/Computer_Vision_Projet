import os
import pandas as pd
import rasterio
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, default_collate
import re

###############################################################################
#                  FONCTION COLLATE POUR IGNORER LES ÉCHANTILLONS None
###############################################################################
def skip_none_collate_fn(batch):
    """
    Filtre les éléments None avant de les assembler en batch.
    Si tous les éléments du batch sont None, renvoie None.
    """
    filtered_batch = [x for x in batch if x is not None]
    print(f"Batch après filtrage: {len(filtered_batch)} éléments")  # Debug
    return None if len(filtered_batch) == 0 else torch.utils.data.dataloader.default_collate(filtered_batch)


###############################################################################
#                  FONCTION UTILITAIRE POUR MODIFIER UNIQUEMENT LA DATE
###############################################################################
def replace_date_in_path(path):
    """
    Repère une date au format YYYY-MM-DD.tif dans un chemin et génère 
    une version où la date est remplacée par YYYY_MM_DD.tif.

    Si aucune date n'est trouvée, retourne le chemin d'origine.
    """
    date_pattern = re.compile(r"(\d{4})-(\d{2})-(\d{2})\.tif$")
    
    match = date_pattern.search(path)
    if match:
        date_hyphen = match.group(0)  # "YYYY-MM-DD.tif"
        date_underscore = date_hyphen.replace("-", "_")  # "YYYY_MM_DD.tif"
        return path.replace(date_hyphen, date_underscore)
    return path  # Retourne le chemin inchangé s'il n'y a pas de date

###############################################################################
#                         CLASSE FourBandSegDataset
###############################################################################
class FourBandSegDataset(Dataset):
    """
    Dataset pour la segmentation avec 4 canaux d'images (RGB+IR) et un masque.
    Seules les lignes avec alignment == True sont utilisées.
    """

    def __init__(self, dataframe, transform=None):
        df_filtered = dataframe[dataframe['alignment'] == True].copy().reset_index(drop=True)
        self.df = df_filtered
        self.transform = transform  # Transformations éventuelles

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Récupération des chemins
        planet_path_hyphen = row.get("planet_path", "")
        labels_path_hyphen = row.get("labels_path", "")

        # Générer leur version avec underscore UNIQUEMENT POUR LA DATE
        planet_path_underscore = replace_date_in_path(planet_path_hyphen)
        labels_path_underscore = replace_date_in_path(labels_path_hyphen)

        # Vérifier l'existence des fichiers
        planet_path = planet_path_hyphen if os.path.exists(planet_path_hyphen) else planet_path_underscore
        label_path = labels_path_hyphen if os.path.exists(labels_path_hyphen) else labels_path_underscore

        # Vérification finale : on ignore si aucun fichier n'existe
        if not os.path.exists(planet_path) or not os.path.exists(label_path):
            return None

        # Lecture de l'image (4 canaux)
        try:
            with rasterio.open(planet_path) as src:
                image = src.read()  # [4, H, W]
        except:
            return None

        # Lecture du masque
        try:
            with rasterio.open(label_path) as src:
                label = src.read(1)  # [H, W]
        except:
            return None
        
        # Conversion en tenseurs PyTorch
        image_tensor = torch.from_numpy(image).float()  # [4, H, W]
        label_tensor = torch.from_numpy(label).long()   # [H, W]

        # (Optionnel) transform / augmentation
        if self.transform is not None:
            pass

        return image_tensor, label_tensor

###############################################################################

