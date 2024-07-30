from georefe_ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QFileDialog
import open3d as o3d

import os
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        """
        Initialise une instance de la classe MyMainWindow.

        Appelle le constructeur de la classe parent MyMainWindow et initialise l'interface utilisateur.

        Attributes:
            path_lidar_brute (str): Chemin du fichier LIDAR brut.
            dgps_path (str): Chemin du fichier DGPS.
            lidar_geo_path (str): Chemin du fichier LIDAR géoréférencé.
            lidar_sortedByProfile_line: LIDAR trié par ligne de profil.
            lidar_data_geo (list): Liste des données LIDAR géoréférencées.
            Bras_levier (numpy.ndarray): Vecteur représentant le bras de levier.
            Matrice_rot (numpy.ndarray): Matrice de rotation.
        """
        super(MyMainWindow, self).__init__()
        self.setupUi(self)
        self.initialize_ui()

    def initialize_ui(self):
        """
        Initialise les attributs de l'interface utilisateur.
        """
        self.path_lidar_brute = None
        self.dgps_path = None
        self.lidar_geo_path = None
        self.lidar_sortedByProfile_line = None
        self.lidar_data_geo = []
        self.Bras_levier = np.transpose(np.array([0.14, 0.249, -0.076]))
        self.Matrice_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.connect_buttons()

    def connect_buttons(self):
        """
        Connecte les boutons de l'interface utilisateur aux méthodes correspondantes.

        - path_lidar_bf_geo_btn : Importe les données LIDAR brut.
        - dgps_path_btn : Importe les données DGPS.
        - lidar_geo_path_btn : Définit le chemin de sortie pour les données LIDAR géoréférencées.
        - georeferencer_btn : Calcule le géoréférencement des données LIDAR.
        - visualiser_btn : Lance la visualisation des données LIDAR.
        """
        self.path_lidar_bf_geo_btn.clicked.connect(self.import_lidarData_brute)
        self.dgps_path_btn.clicked.connect(self.import_dgps)
        self.lidar_geo_path_btn.clicked.connect(self.set_output_geo_path)
        self.georeferencer_btn.clicked.connect(self.calcule_georeferencement)
        self.visualiser_btn.clicked.connect(self.visualisation_lidar)

    def import_lidarData_brute(self):
        """
        Ouvre une boîte de dialogue pour sélectionner un dossier contenant les données LIDAR brut.

        Met à jour le chemin du dossier LIDAR brut, affiche le chemin dans l'interface utilisateur,
        et imprime le chemin sélectionné dans la console.

        Returns:
            None
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Lidar Data Folder")
        if folder_path:
            self.path_lidar_brute = folder_path
            print(f"Selected Lidar Data Folder: {self.path_lidar_brute}")
            self.path_lidar_bf_geo_txt.setPlainText(self.path_lidar_brute)

    def import_dgps(self):
        """
        Ouvre une boîte de dialogue pour sélectionner un fichier DGPS.

        Met à jour le chemin du fichier DGPS, affiche le chemin dans l'interface utilisateur,
        et imprime le chemin sélectionné dans la console.

        Returns:
            None
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select DGPS File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.dgps_path = file_path
            print(f"Selected DGPS File: {self.dgps_path}")
            self.dgps_path_txt.setPlainText(self.dgps_path)

    def set_output_geo_path(self):
        print(
            "SVP, le choix du répertoire de fichier de sortie ne doit pas être dans le même répertoire que celui du DGPS ni que celui des fichiers LIDAR.")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Output Geo File", "", "XYZ Files (*.xyz);;All Files (*)")
        if file_path:
            self.lidar_geo_path = file_path
            print(f"Selected Output Geo File: {self.lidar_geo_path}")
            self.lidar_geo_path_txt.setPlainText(self.lidar_geo_path)

    def rotation_matrix(self, roll, pitch, heading):
        """
        Calcule et retourne une matrice de rotation 3x3 à partir des angles de roulis, tangage et cap.

        Args:
            roll (float): Angle de roulis en degrés.
            pitch (float): Angle de tangage en degrés.
            heading (float): Angle de cap en degrés.

        Returns:
            numpy.ndarray: Matrice de rotation 3x3 résultante.
        """
        # Convertit les degrés en radians
        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        heading_rad = np.deg2rad(heading)

        # Matrices de rotation élémentaires
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll_rad), -np.sin(roll_rad)],
                       [0, np.sin(roll_rad), np.cos(roll_rad)]])

        Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                       [0, 1, 0],
                       [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

        Rz = np.array([[np.cos(heading_rad), -np.sin(heading_rad), 0],
                       [np.sin(heading_rad), np.cos(heading_rad), 0],
                       [0, 0, 1]])

        # La matrice de rotation finale en combinant les rotations
        R = Rz @ Ry @ Rx

        return R

    def calcule_georeferencement(self):
        """
        Calcule le géoréférencement des données LIDAR à partir des fichiers LIDAR brut et DGPS.

        - Pour chaque fichier XYZ LIDAR brut, les données sont triées par ligne de profil.
        - Les points LIDAR sont géoréférencés en utilisant les données DGPS correspondantes.

        Returns:
            None
        """
        if self.path_lidar_brute is not None and len(self.path_lidar_brute) != 0:
            xyz_files = [file for file in os.listdir(self.path_lidar_brute) if file.endswith(".xyz")]

            # Liste pour stocker les DataFrames pour chaque fichier
            lidarData_byProfile_line = []

            # Itérer sur chaque fichier XYZ
            for xyz_file in xyz_files:
                input_file_path = os.path.join(self.path_lidar_brute, xyz_file)

                # Appliquer la fonction de tri à chaque fichier XYZ et ajouter le résultat à la liste
                lidarData_byProfile_line.append(self.sort_lidar_data_byFile(input_file_path))

            print("finished sorting")
            print("start determining nbr of profiles for each File")

            # c'est l'étape la plus compliquée, puisque nous avons 10 fichiers LIDAR et 1 seul fichier DGPS
            # nous devons parcourir le fichier DGPS au premier temps de 1 jusqu'à 1+nbr des profils dans le fichier
            # pour le même fichier DGPS et le deuxième fichier LIDAR, nous devons commencer maintenant de la ligne où nous nous sommes arrêtés
            # jusqu'à la ligne où nous nous sommes arrêtés plus le nombre de profils dans le deuxième fichier ..etc

            start_end_point_in_dgps_file = self.determinerStart_endPoint(lidarData_byProfile_line)
            print("End determining nbr of profiles for each File")

            for key, value in start_end_point_in_dgps_file.items():
                try:
                    print("start georeferencing file ", key)
                    with open(self.dgps_path, 'r') as file:
                        lines = file.readlines()
                        # on commence par les points lidar ayant un profile numero 1 , puis on increment , jusqu a
                        # on arrive a la ligne qui correspond dans le fichier dgps au dernier profile dans le fichier lidar
                        indexProfile = 0
                        for line in lines[value['start_point']:value['end_point'] + 1]:
                            data = line.strip().split('\t')
                            X_DGPS = float(data[1])  # Convert X_GPS to a float , deuxième colonne
                            Y_DGPS = float(data[2])  # Convert Y_GPS to a float, troisième colonne
                            Z_DGPS = float(data[3])  # Convert Z_GPS to a float, quatrième colonne
                            Roll = float(data[7])  # Convert Roll to a float, colonne n 8
                            Pitch = float(data[8])
                            Heading = float(data[9])
                            rotationMatrice = (self.rotation_matrix(Roll, Pitch, Heading))
                            # Print rows where the second column is equal to 0 from the first_file_data DataFrame
                            LidarBruteByFile = lidarData_byProfile_line[key]
                            rows_same_profile = LidarBruteByFile[LidarBruteByFile[1] == indexProfile]
                            if not rows_same_profile.empty:
                                xyz_lidar = rows_same_profile.iloc[:, 2:5]
                                for index, row in xyz_lidar.iterrows():
                                    # Accéder à chaque valeur de la ligne
                                    x_lidar = row[2]
                                    y_lidar = row[3]
                                    z_lidar = row[4]
                                    lidar_vecteur = np.array([x_lidar, y_lidar, z_lidar])
                                    # établir la relation du cours
                                    georeferenced_point = np.array([X_DGPS, Y_DGPS, Z_DGPS]) + rotationMatrice @ (
                                            self.Bras_levier + self.Matrice_rot @ lidar_vecteur)
                                    self.lidar_data_geo.append(georeferenced_point)
                            indexProfile = indexProfile + 1
                        print("end georeferencing file ", key)

                except FileNotFoundError:
                    print(f"File '{self.dgps_path}' not found.")
                except Exception as e:
                    print(f"An error occurred: {e}")
            print("nombre de points total : ",len(self.lidar_data_geo))

            print("start saving")
            self.saveLidarToFile()
            print("end saving")
            msg = QMessageBox()
            msg.setWindowTitle("Succès")
            msg.setText("Les fichiers LiDAR ont été géoréférencés avec succès,Pour les visualiser, cliquer sur visualiser.")
            msg.exec_()
            print("you can visualise it now by clicking on : visualiser")
        else:
            print("SVP importer vos fichiers correctement !")

    def sort_lidar_data_byFile(self, input_file_path):
        """
        Trie les données LIDAR d'un fichier XYZ d'abord par profil, puis par ligne.

        Args:
            input_file_path (str): Chemin du fichier XYZ LIDAR brut.

        Returns:
            pandas.DataFrame: DataFrame trié par profil puis par ligne.
        """
        print("sorting by profile first then by line, for file:", input_file_path)

        # Charger les données dans un DataFrame
        df = pd.read_csv(input_file_path, sep=' ', header=None)

        # Supposer que les deux premières colonnes représentent 'line' et 'profile' pour le tri
        df[0] = pd.to_numeric(df[0], errors='coerce')
        df[1] = pd.to_numeric(df[1], errors='coerce')

        # Supprimer les lignes où ces colonnes n'ont pas pu être converties en nombres
        df = df.dropna(subset=[0, 1])

        # Tri du DataFrame d'abord par la deuxième colonne (profil), puis par la première colonne (ligne)
        sorted_df = df.sort_values(by=[1, 0])

        return sorted_df

    def determinerStart_endPoint(self, lidarData_byProfile_line):
        """
        Détermine les points de départ et de fin pour chaque fichier en fonction du nombre de profils.

        Args:
            lidarData_byProfile_line (list): Liste de DataFrames triés par profil puis par ligne.

        Returns:
            dict: Dictionnaire contenant les points de départ et de fin pour chaque fichier.
        """
        print("start determining start and end point for each file")

        n_profileByFile = []
        for nn, sorted_df in enumerate(lidarData_byProfile_line):
            last_line_second_column = sorted_df.iloc[-1, 1]
            n_profileByFile.append(last_line_second_column)
        start_end_point_in_dgps_file = {}
        start_point = 1
        # compteur correspond au nombre de fichiers dans le répertoire, si 0 fichier, 1
        compteur = 0
        for value in n_profileByFile:
            end_point = start_point + value
            start_end_point_in_dgps_file[compteur] = {'start_point': start_point, 'end_point': end_point}
            start_point = end_point + 1
            compteur = compteur + 1
        print("finish determining start and end point for each file")
        return start_end_point_in_dgps_file

    def saveLidarToFile(self):
        """
        Sauvegarde les points géoréférencés dans un fichier texte en format xyz.

        Le fichier est enregistré au chemin spécifié par l'attribut 'lidar_geo_path'.

        Returns:
            None
        """
        # Sauvegarder les points géoréférencés dans un fichier texte
        np.savetxt(self.lidar_geo_path, self.lidar_data_geo, fmt="%.8f", delimiter=" ")
        print(f"Georeferenced points saved to {self.lidar_geo_path}")

    def visualisation_lidar(self):
        """Visualise les données LiDAR si disponibles."""

        # Vérification si des points LiDAR sont disponibles
        if len(self.lidar_data_geo) != 0:
            print("En cours")
            # Création d'un objet PointCloud à partir des coordonnées sélectionnées
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.lidar_data_geo)
            print("Affichage avec succès")
            # Visualisation de la nuage de points en 3D à l'aide d'Open3D
            o3d.visualization.draw_geometries([pcd])
        else:
            # Affichage d'un message si aucune donnée LiDAR n'est disponible
            print("Aucune donnée LiDAR disponible.")


# Création de l'application
app = QtWidgets.QApplication([])

# Création d'une instance de votre MainWindow personnalisée
main_window = MyMainWindow()

# Affichage de la fenêtre principale
main_window.show()

# Démarrage de la boucle d'événements de l'application
import sys
sys.exit(app.exec_())


