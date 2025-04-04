import os
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

class LabelsCharacteristics:
    def __init__(self, base_path):
        """
        Initializes the Class with the path to the folder with the labels 
        dataset
        Arguments : 
        base_pah : path to the folder
        """
        self.base_path = base_path

    def list_zones(self, base_path):
        """
        Lists the zones of interests using the labels dataset

        Arguments : base_path, path of the folder containing the labels dataset

        Returns : list ot the zones of interest, named by the name of each
        folder
        """
        # dossier à parcourir
        dossier = base_path

        #liste pour stocker les noms de sous-dossiers
        zones_interet = []

        # Parcourir chaque élément dans le dossier
        for element in os.listdir(dossier):
            chemin_complet = os.path.join(dossier, element)
            if os.path.isdir(chemin_complet):
                zones_interet.append(element)

        # Afficher la liste des sous-dossiers
        print(zones_interet)
        return zones_interet
    
    
    def get_lulc_per_zone_auto(self, base_path, zones_interest):
        """
        Gets for each zone of interest the LULC class present in the dataset,
        over the 2 years of data acquisition, and the number of classes.

        Arguments:
            - base_path: path of the folder containing the labels dataset
            - zones_interest: list of the zone of interest

        Returns:
        a dataframe with the classes as columns, the zones as rows, and a 1 if
        the class is present, a 0 otherwise, the last column is the number of
        classes present in the zone of interest
        """
        # Collecter les noms des classes LULC
        lulc = set()
        for zone in zones_interest:
            zone_path = os.path.join(base_path, zone)
            labels_vector_path = os.path.join(zone_path, 'Labels', 'Vector')
            if os.path.isdir(labels_vector_path):
                for sous_dossier in os.listdir(labels_vector_path):
                    chemin_complet = os.path.join(labels_vector_path, sous_dossier)
                    if os.path.isdir(chemin_complet):
                        lulc.add(sous_dossier)

        # Convertir le set en liste
        lulc = list(lulc)

        # Initialiser un dictionnaire pour stocker les résultats
        resultats = {zone: {nom: 0 for nom in lulc} for zone in zones_interest}

        # Parcourir chaque dossier
        for zone in zones_interest:
            zone_path = os.path.join(base_path, zone)
            labels_vector_path = os.path.join(zone_path, 'Labels', 'Vector')
            if os.path.isdir(labels_vector_path):
                # Parcourir chaque sous-dossier dans Labels/Vector
                for sous_dossier in os.listdir(labels_vector_path):
                    chemin_complet = os.path.join(labels_vector_path, sous_dossier)
                    if os.path.isdir(chemin_complet) and sous_dossier in lulc:
                        resultats[zone][sous_dossier] = 1

        # Convertir le dictionnaire en DataFrame
        df = pd.DataFrame(resultats).T
        df['class_number'] = df.sum(axis=1)

        # Afficher le DataFrame
        return df



    def analyse_lulc_per_zone(self, df):
        """
        Computes for the set of zones of interest : the number of zones, max
        and min number of classes per zone, and occurence of each class

        Argument : df containing the classes and sum of classes as columns and
        zones of interest as rows

        Returns : 
        kpis : list containing the number of zones, min number of classes and 
        max number of classes 
        occurences : data frame with the occurence of each class as column and
        the classes names as rows
        """
        # Calculer le nombre de lignes du DataFrame
        nombre_de_lignes = len(df)
        max_classes=df['class_number'].max()
        min_classes=df['class_number'].min()

        # Calculer l'occurrence de chaque colonne
        occurrences = df.drop(columns=['class_number']).apply(lambda x: x.sum())

        # Afficher les résultats
        print(f"Nombre de lignes: {nombre_de_lignes}")
        print("Occurrences de chaque colonne:")
        print(occurrences)
        print(f'le nombre max de classes est {max_classes}')
        print(f'le nombre min de classes est {min_classes}')

        kpis = [nombre_de_lignes, min_classes, max_classes]

        return kpis, occurrences

    def plot_images_per_zone(self, base_path, zones_interest):
        for zone in zones_interest:
            zone_path = os.path.join(base_path, zone)
            labels_raster_path = os.path.join(zone_path, 'Labels', 'Raster')
            if os.path.isdir(labels_raster_path):
                image_2018 = None
                image_2019 = None
                for root, dirs, files in os.walk(labels_raster_path):
                    for file in files:
                        if file.endswith('2018_01_01.tif'):
                            image_2018 = os.path.join(root, file)
                        elif file.endswith('2019_12_01.tif'):
                            image_2019 = os.path.join(root, file)
                
                if image_2018 and image_2019:
                    fig, axes = plt.subplots(2, 7, figsize=(15, 10))
                    fig.suptitle(f'Zone: {zone}', fontsize=16)
                    with rasterio.open(image_2018) as src:
                        for i in range(7):
                            show(src.read(i + 1), ax=axes[0, i])
                            #axes[0, i].set_title(f'{zone} - 2018_01_01 - Band {i + 1}', fontsize=8)
                    with rasterio.open(image_2019) as src:
                        for i in range(7):
                            show(src.read(i + 1), ax=axes[1, i])
                            #axes[1, i].set_title(f'{zone} - 2019_12_01 - Band {i + 1}', fontsize=8)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
                    plt.show()
        
    
def main():
    # base path for the train data set
    base_path = '/Users/fabreindira/Library/CloudStorage/OneDrive-telecom-paristech.fr/MS_BGD/BGDIA_706_Airbus/dataset/labels'
    base_path_test='/Users/fabreindira/Library/CloudStorage/OneDrive-telecom-paristech.fr/MS_BGD/BGDIA_706_Airbus/dataset/dynamicearthnet_test_labels'
    lc = LabelsCharacteristics(base_path)
    zones = lc.list_zones(base_path)
    zones_test = lc.list_zones(base_path_test)
    print(zones)

    df = lc.get_lulc_per_zone_auto(base_path, zones)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    df_test = lc.get_lulc_per_zone_auto(base_path_test, zones_test)
    print(tabulate(df_test, headers='keys', tablefmt='psql'))

    nombre_de_lignes, occurrences = lc.analyse_lulc_per_zone(df)
    nombre_de_lignes_test, occurrences_test = lc.analyse_lulc_per_zone(df_test)


if __name__ == "__main__":
    main()
    