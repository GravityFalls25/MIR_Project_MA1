# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'parite2_ui.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!

from flickr30k import Dataset #pour avoir dataset partie3


from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import operator, math, os, glob
import torch.nn as nn
from matplotlib.pyplot import imread

import timm
from torchvision import transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore

from sentence_transformers import SentenceTransformer

import json

import clip
import faiss


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # self.textInput = QtWidgets.QTextEdit(self.centralwidget)
        # self.textInput.setGeometry(QtCore.QRect(60, 80, 321, 31))
        # self.textInput.setObjectName("textInput")

        # self.topInput = QtWidgets.QTextEdit(self.centralwidget)
        # self.topInput.setGeometry(QtCore.QRect(60, 40, 321, 31))
        # self.topInput.setObjectName("topInput")

        # First input: textInput (on the left)
        self.textInput = QtWidgets.QTextEdit(self.centralwidget)
        self.textInput.setGeometry(QtCore.QRect(20, 80, 321, 31))  # X=20
        self.textInput.setObjectName("textInput")

        # Second input: topInput (on the right of textInput)
        self.topInput = QtWidgets.QTextEdit(self.centralwidget)
        self.topInput.setGeometry(QtCore.QRect(351, 80, 30, 31))  # X = 20 + 321 + 10
        self.topInput.setObjectName("topInput")
        self.topInput.setText("10")  # default value

        #selecteur de partie pour avoir soir la 2 ou la 3
        self.partSelector = QtWidgets.QComboBox(self.centralwidget)
        self.partSelector.setGeometry(QtCore.QRect(351 + 30 + 5, 80, 100, 31))  # x = 351+30+5 = 386
        self.partSelector.setObjectName("partSelector")
        self.partSelector.addItems(["partie 2", "partie 3"])


        self.titreLabe = QtWidgets.QLabel(self.centralwidget)
        self.titreLabe.setGeometry(QtCore.QRect(270, 30, 211, 16))
        self.titreLabe.setAlignment(QtCore.Qt.AlignCenter)
        self.titreLabe.setObjectName("titreLabe")
        self.loadImage = QtWidgets.QPushButton(self.centralwidget)
        self.loadImage.setGeometry(QtCore.QRect(570, 230, 161, 23))
        self.loadImage.setObjectName("loadImage")
        self.research = QtWidgets.QPushButton(self.centralwidget)
        self.research.setGeometry(QtCore.QRect(60, 120, 80, 23))
        self.research.setObjectName("research")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(70, 190, 461, 351))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 459, 349))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.scroll_layout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollAreaWidgetContents.setLayout(self.scroll_layout)

        self.map = QtWidgets.QLabel(self.centralwidget)
        self.map.setGeometry(QtCore.QRect(70, 550, 461, 30))  # x, y, width, height
        self.map.setObjectName("map")
        self.map.setText("")  # Initially empty
        self.map.setAlignment(QtCore.Qt.AlignCenter)

        self.imageCharge = QtWidgets.QLabel(self.centralwidget)
        self.imageCharge.setGeometry(QtCore.QRect(550, 80, 221, 131))
        self.imageCharge.setText("")
        self.imageCharge.setObjectName("imageCharge")
        self.coubreRP = QtWidgets.QLabel(self.centralwidget)
        self.coubreRP.setGeometry(QtCore.QRect(550, 270, 300, 300))
        self.coubreRP.setObjectName("coubreRP")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        

        #charge le modèle ViT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #prend GPU si disponible sinon cpu
        #charge modèle ViT pour avoir un    vision transformer
        self.feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.feature_extractor.reset_classifier(0)  # Remove classification head to get pure features
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.model_text = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') #import le modele

        self.features_dicts, self.image_dict, self.text_dict = self.load_features_with_images()

        self.selected_image = None

        #chargement partie 3 commence par le dataset
        builder = Dataset()
        builder.download_and_prepare() 

        ds = builder.as_dataset()
        self.dataset = ds['test']

        self.max_char_len = 200 #car si dépasse 77 token modèle plante donc limite à 200 mots pour être tranquille

        self.model_clip, self.preprocess = clip.load("ViT-B/32", device=self.device) #charge le modèle CLIP

        #charge des index par faiss 
        self.image_index = faiss.read_index("flickr30k_clip_images.index")
        self.text_index = faiss.read_index("flickr30k_clip_texts.index")
        self.multimodal_index = faiss.read_index("flickr30k_clip_multimodal.index")
        self.image_paths = np.load("image_paths.npy", allow_pickle=True)
        self.captions = np.load("captions.npy", allow_pickle=True)




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.titreLabe.setText(_translate("MainWindow", "Recherche multimodale"))
        self.loadImage.setText(_translate("MainWindow", "Charger une image"))
        self.research.setText(_translate("MainWindow", "Rechercher"))
        self.coubreRP.setText(_translate("MainWindow", "Courbre R/P"))
        self.loadImage.clicked.connect(self.Ouvrir)
        self.research.clicked.connect(self.recherche)


    def Ouvrir(self, MainWindow): 
        global fileName 
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpeg *.jpg *.bmp)") 
        pixmap = QtGui.QPixmap(fileName) 
        pixmap = pixmap.scaled(self.imageCharge.width(), 
        self.imageCharge.height(), QtCore.Qt.KeepAspectRatio) 
        self.imageCharge.setPixmap(pixmap) 
        self.imageCharge.setAlignment(QtCore.Qt.AlignCenter) 
        self.selected_image = fileName #stock l'image sélectionnée 

    def extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(image)
        return features.cpu().numpy().squeeze().flatten()

    def euclidean_distance(self, vec1, vec2) :
        distance = np.linalg.norm(vec1 - vec2)
        return distance

    def getkVoisins(self, features_dict, querry, k=5):
        """
        Find k nearest neighbors to a query (by name or vector) in the given features_dict.

        Parameters:
            features_dict: dict {image_name: feature_vector}
            query_name: the name of the image in the dict (optional if query_vector is provided)
            query_vector: the query vector (optional if query_name is provided)
            k: number of neighbors to retrieve

        Returns:
            list of (name, distance) tuples
        """
        
        distances = []
        for name, feature_vector in features_dict.items():
            dist = self.euclidean_distance(querry, feature_vector)
            distances.append((name, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]


    #charge les données
    def load_features_with_images(self):
        feature_folder = "ViT_descriptor"
        text_feature_folder = "text_descriptor"
        image_folder = "MIR_DATASETS_B/MIR_DATASETS_B"
        # Récupère tous les fichiers .txt dans tous les sous-dossiers
        feature_files = sorted(glob.glob(os.path.join(feature_folder, "**", "*.npy"), recursive=True))
        feature_text_files = sorted(glob.glob(os.path.join(text_feature_folder, "**", "*.npy"), recursive=True))
        feature_text_dict = {}
        features_dict = {}
        image_dict = {}

        text_dict = {}

        with open('captions_MIR_DATASETS_B.json', 'r') as f:
            text_complet = json.load(f)

        for file in feature_files:
            feature_vector = np.load(file)
            base_name = os.path.splitext(os.path.basename(file))[0]

            # Stocker les features
            features_dict[base_name] = feature_vector

            # Recherche de l'image correspondante (même sous-structure que features)
            rel_path = os.path.relpath(file, feature_folder)  # ex: classe/sub_classe/nom.txt
            # print(rel_path)
            rel_image_path = os.path.splitext(rel_path)[0] 
            # print(rel_image_path)
            rel_image_path = rel_image_path + ".jpg"  # classe/sub_classe/nom.jpg
            # print(rel_image_path)
            image_path = os.path.join(image_folder, rel_image_path)

            text_dict[base_name] = text_complet[rel_image_path]

            if os.path.exists(image_path):
                image_dict[base_name] = image_path
            else:
                print(f"Aucune image trouvée pour {file} (attendu : {image_path})")
        
        for file in feature_text_files:
            feature_vector = np.load(file)
            base_name = os.path.splitext(os.path.basename(file))[0]

            # Stocker les features
            feature_text_dict[base_name] = feature_vector

            # Recherche de l'image correspondante (même sous-structure que features)
            rel_path = os.path.relpath(file, feature_folder)  # ex: classe/sub_classe/nom.txt
            # print(rel_path)
            rel_image_path = os.path.splitext(rel_path)[0] 
            # print(rel_image_path)
            rel_image_path = rel_image_path+ ".jpg"  # classe/sub_classe/nom.jpg
            # print(rel_image_path)
            image_path = os.path.join(image_folder, rel_image_path)

            # if os.path.exists(image_path):
            #     if base_name not in image_dict:
            #         image_dict[base_name] = image_path #ajote si connait pas
            #         print("added new image")
            # else:
            #     print(f"Aucune image trouvée pour {file} {image_path}")

        print(f"{len(features_dict)} caractéristiques chargées avec images depuis {feature_folder}")
        print(f"{len(feature_text_dict)} caractéristiques chargées pour text avec images depuis {text_feature_folder}")

        features_mixed_dict = {}
        for key in features_dict:
            if key in feature_text_dict:
                features_mixed_dict[key] = np.hstack([features_dict[key]/np.linalg.norm(features_dict[key]), feature_text_dict[key]/np.linalg.norm(feature_text_dict[key])]) #mets image puis text pour faire la recherche multimodal

        first_key = next(iter(features_mixed_dict))
        shape = features_mixed_dict[first_key].shape
        

        return (features_dict, feature_text_dict, features_mixed_dict), image_dict, text_dict


    def compute_metrics(self, retrieved_indices, relevant_indices, k=None):
   
        if k is not None:
            retrieved_indices = retrieved_indices[:k]
        relevant_set = set(relevant_indices)
        retrieved_set = set(retrieved_indices)
        num_relevant = len(relevant_set)
        if num_relevant == 0:
            return 0.0, 0.0, 0.0


        precisions = []
        recalls = []
        recalls_top_k = []

        
        hits = 0
        sum_precisions = 0.0
        for i, idx in enumerate(retrieved_indices):
            if idx in relevant_set:
                hits += 1
                sum_precisions += hits / (i + 1)
                precision = hits / (i + 1)
                recall = hits / num_relevant
                recall_top_k = hits / k
                precisions.append(precision)
                recalls.append(recall)
                recalls_top_k.append(recall_top_k)
        average_precision = sum_precisions / num_relevant if num_relevant > 0 else 0.0

        return precisions, recalls, recalls_top_k, average_precision


    #fonction pour afficher un résultat avec le nom au dessus et le texte en dessous
    def add_result_image_and_text(self, scroll_area_widget, image_path, top_text, bottom_text):
        # Create a container widget for each result
        result_widget = QWidget()
        layout = QVBoxLayout()
        result_widget.setLayout(layout)

        # Top text label
        label_top = QLabel(top_text)
        label_top.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label_top)

        # Image label
        label_image = QLabel()
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(200, 200, QtCore.Qt.KeepAspectRatio)  # Optional resize
        label_image.setPixmap(pixmap)
        label_image.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label_image)

        # Bottom text label
        label_bottom = QLabel(bottom_text)
        label_bottom.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label_bottom)

        # Add this result widget to your scrollAreaWidgetContents layout
        scroll_area_widget.layout().addWidget(result_widget)
    
    def recherche(self):
        #reset les résultats précédents
        self.coubreRP.clear()  # reset courbeRP
        self.coubreRP.setText("Courbe R/P à afficher ici")  

        for i in reversed(range(self.scrollAreaWidgetContents.layout().count())): #supprime les ancienes images
            widget_to_remove = self.scrollAreaWidgetContents.layout().itemAt(i).widget()
            widget_to_remove.setParent(None)
    
        if self.partSelector.currentText() == "partie 2":
            self.recherche_partie2()
        elif self.partSelector.currentText() == "partie 3":
            self.recherche_partie3()
        else:
            print("Partie non reconnue, veuillez sélectionner partie 2 ou partie 3.")

    
    def recherche_partie3(self): #fonction recherche pour la partie 3
        input_text = self.textInput.toPlainText()
        top_text = self.topInput.toPlainText()
        image_path = self.selected_image
        print("image_path", image_path)

        if image_path == "":
            image_path = None

        assert top_text != "", "Veuillez entrer un nombre d'image souhaite"

        top = int(top_text) #le récupère
        assert top > 0, "Veuillez entrer un nombre d'image souhaite positif"

        #calcule les embeddings de l'image si elle est sélectionnée
        selected_request = -1
        image_embedding = None
        text_embedding = None

        if image_path is not None:
            selected_request += 1
            query_image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_embedding = self.model_clip.encode_image(query_image).cpu().numpy()
    

        if input_text is not None:
            selected_request += 2
            text_token = clip.tokenize([input_text[:self.max_char_len]]).to(self.device)  # Tokenize the input text
            with torch.no_grad():
                text_embedding = self.model_clip.encode_text(text_token).cpu().numpy()

        #cree embedding final + rechecheche
        if image_embedding is not None and text_embedding is None:
            D, I = self.image_index.search(image_embedding.astype('float32'), k=top)  # Recherche dans l'index d'images
        elif image_embedding is None and text_embedding is not None:
            D, I = self.text_index.search(text_embedding.astype('float32'), k=top)
        else:
            embedding = np.hstack([image_embedding/np.linalg.norm(image_embedding), text_embedding/np.linalg.norm(text_embedding)]) #concatène les deux embeddings
            print("shape text_embedding", text_embedding.shape)
            print("shape image_embedding", image_embedding.shape)
            print("shape embedding", embedding.shape)
            D, I = self.multimodal_index.search(embedding.astype('float32'), k=top)  # Recherche dans l'index multimodal

        top_image_paths = [self.image_paths[i] for i in I[0]]

        # nom_images_proches = [self.image_dict[v[0]] for v in voisins if v[0] in self.image_dict]
        for j in range(min(top, len(top_image_paths))):
            image_path_tmp = top_image_paths[j]
            indice = np.where(self.image_paths == image_path_tmp)[0]  # Trouve l'indice de l'image dans self.image_paths
            print("indice", indice)
            nom = top_image_paths[j].split("/")[-1].split(".")[0]  #retire extension + path avant
            # print(self.text_dict)
            text = self.captions[indice[0]] if indice.size > 0 else "Texte non trouvé"
            text = "\n ".join([str(t) for t in text.tolist()]) #pour avoir un seul string
            print("text", text)
            
            self.add_result_image_and_text(self.scrollAreaWidgetContents, image_path_tmp, nom, text)
        
        if selected_request == 1: #si que fait requête texte, calcule les métriques car sinon ne fait pas de sens
            query_words = set(input_text.lower().split())
            relevant_indices = []

            
            for idx, img_captions in enumerate(self.captions):  # now a list of 5 strings
                for cap in img_captions:
                    cap_words = set(cap.lower().split())
                    if query_words.issubset(cap_words):
                        relevant_indices.append(idx)
                        break  # only need one matching caption to count the image as relevant
        
            retrieved_indices = [self.image_paths.tolist().index(p) for p in top_image_paths]
            # print("retrieved_indices", retrieved_indices)
            # print("relevant_indices", relevant_indices)
            precision, recall, recall_top_k, average_precision = self.compute_metrics(retrieved_indices, relevant_indices, k=top)
            #plot la courbe
            plt.figure(figsize=(4, 4))
            

            plt.plot(recall, precision, color='red', label='Courbe Rappel/Précision TOPMAX')
            plt.plot(recall_top_k, precision, color='blue', label='Courbe Rappel/Précision TOP K')
            plt.title("Courbe Rappel/Précision")

            plt.xlabel("Rappel")
            plt.ylabel("Precision")
            plt.ylim(-0.1, 1.1)
            plt.xlim(-0.1, 1.1)
            plt.grid()

            plt.legend(loc='best')
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()  

            # Load buffer as QPixmap
            buf.seek(0)
            qimg = QtGui.QImage.fromData(buf.getvalue())
            pixmap = QtGui.QPixmap.fromImage(qimg)

            # Set pixmap to your QLabel
            self.coubreRP.setPixmap(pixmap)
            self.coubreRP.setScaledContents(True)

            self.map.setText(f"MAP = {average_precision:.2f}")
            
        
            


    def recherche_partie2(self): #fonction recherche pour la partie 2

        race = True

        #les inputs
        input_text = self.textInput.toPlainText()
        top_text = self.topInput.toPlainText()
        image_path = self.selected_image
        print("image_path", image_path)

        if image_path == "":
            image_path = None

        assert top_text != "", "Veuillez entrer un nombre d'image souhaite"

        top = int(top_text) #le récupère
        assert top > 0, "Veuillez entrer un nombre d'image souhaite positif"

        if input_text == "":
            input_text = None
        #regarde ce qui est rempli pour voir quel type de requête doit faire
        selected_request = -1
        image_embedding = None
        text_embedding = None

        if image_path is not None:
            selected_request += 1
            print("image_path", image_path)
            image_embedding = self.extract_features(image_path)  #récupère les features de l'image requête

        if input_text is not None:
            selected_request += 2
            print("input_text", input_text)
            text_embedding = self.model_text.encode(input_text)  # recupère l'embeding du text

        #prend en comtpte si ils n ont pas les même dimensions
        if image_embedding is not None and text_embedding is None:
            embedding = image_embedding
        elif image_embedding is None and text_embedding is not None:
            embedding = text_embedding
        else:
            embedding = np.hstack([image_embedding/np.linalg.norm(image_embedding), text_embedding/np.linalg.norm(text_embedding)]) #concatène les deux embeddings

        voisins = self.getkVoisins(self.features_dicts[selected_request], embedding, top) #récupère les k plus proches voisins

        nom_images_proches = [self.image_dict[v[0]] for v in voisins if v[0] in self.image_dict]
        for j in range(min(top, len(nom_images_proches))):
            image_path_tmp = nom_images_proches[j]
            nom = nom_images_proches[j].split("/")[-1].split(".")[0]  #retire extension + path avant
            # print(self.text_dict)
            text = self.text_dict[nom]
            self.add_result_image_and_text(self.scrollAreaWidgetContents, image_path_tmp, nom, text)

        #calcule la courbe R/P
        if selected_request == 0:
            rappel_precion = []
            rp = []
            position1 = os.path.splitext(os.path.basename(image_path))[0]
            
            query_dir = os.path.dirname(image_path) #récupère le nom du dossier de l'image requête
            topmax = len([f for f in os.listdir(query_dir) if os.path.isfile(os.path.join(query_dir, f)) and f.lower().endswith(('.jpg'))])
            

            if race:
                position1 = position1.split("_")[3]
            else:
                position1 = position1.split("_")[2]
            
            for j in range(top):
                position2 = os.path.splitext(os.path.basename(nom_images_proches[j]))[0]

                if race:
                    position2 = position2.split("_")[3]
                else:
                    position2 = position2.split("_")[2]
                
                # print(f"pos {j} same = {position1 == position2}")
                # print("position1", position1)
                # print("position2", position2)

                if position1 == position2:
                    rappel_precion.append("pertinent")
                    # print("pertinent")
                else:
                    rappel_precion.append("non pertinent")
                    # print(j)
                    # print("position1", position1)
                    # print("position2", position2)
                    # print("non pertinent")
            val = 0 #nombre de fois pertinent

            vec_prec = []
            vec_rap = []
            ver_rap_top = []

            for i in range(top):
                if rappel_precion[i] == "pertinent":
                    val += 1
                precision = val/(i+1)
                rappel = val/topmax #bizare comment calculé
                vec_prec.append(precision)
                vec_rap.append(rappel)
                ver_rap_top.append(val/top)
            
            plt.figure(figsize=(4, 4))
            
            # print("vec_prec", vec_prec)
            # print("vec_rap", vec_rap)

            plt.plot(vec_rap, vec_prec, color='red', label='Courbe Rappel/Précision TOPMAX')
            plt.plot(ver_rap_top, vec_prec, color='blue', label='Courbe Rappel/Précision TOP K')
            plt.title("Courbe Rappel/Précision")

            plt.xlabel("Rappel")
            plt.ylabel("Precision")
            plt.ylim(-0.1, 1.1)
            plt.xlim(-0.1, 1.1)
            plt.grid()

            plt.legend()
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()  

            # Load buffer as QPixmap
            buf.seek(0)
            qimg = QtGui.QImage.fromData(buf.getvalue())
            pixmap = QtGui.QPixmap.fromImage(qimg)

            # Set pixmap to your QLabel
            self.coubreRP.setPixmap(pixmap)
            self.coubreRP.setScaledContents(True)




if __name__ == "__main__":
    global filenames
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    
    filenames = os.getcwd() + "/MIR_DATASETS_B/MIR_DATASETS_B" 
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

