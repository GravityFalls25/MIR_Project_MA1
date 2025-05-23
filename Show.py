# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TP3_recherche.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import * 
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import QFileDialog 
import cv2 
import numpy as np 
from skimage.transform import resize 
from skimage.io import imread 
from skimage.feature import hog 
from skimage import exposure 
from matplotlib import pyplot as plt 
from functions import compter_fichiers, extractReqFeatures, showDialog, generateSIFT, generateHistogramme_HSV, generateHistogramme_Color, generateORB 
from distances import * 
import glob

import os

folder_model="" 

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 550)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(120, 10, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setFrameShape(QtWidgets.QFrame.Panel)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.checkBox_HistC = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_HistC.setGeometry(QtCore.QRect(110, 50, 81, 17))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_HistC.setFont(font)
        self.checkBox_HistC.setObjectName("checkBox_HistC")
        self.checkBox_HSV = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_HSV.setGeometry(QtCore.QRect(190, 50, 81, 17))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_HSV.setFont(font)
        self.checkBox_HSV.setObjectName("checkBox_HSV")
        self.checkBox_SIFT = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_SIFT.setGeometry(QtCore.QRect(110, 80, 81, 17))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_SIFT.setFont(font)
        self.checkBox_SIFT.setObjectName("checkBox_SIFT")
        self.checkBox_ORB = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_ORB.setGeometry(QtCore.QRect(190, 80, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_ORB.setFont(font)
        self.checkBox_ORB.setObjectName("checkBox_ORB")
        self.checkBox_GLCM = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_GLCM.setGeometry(QtCore.QRect(190, 110, 81, 17))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_GLCM.setFont(font)
        self.checkBox_GLCM.setObjectName("checkBox_GLCM")
        self.checkBox_LBP = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_LBP.setGeometry(QtCore.QRect(110, 110, 81, 17))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_LBP.setFont(font)
        self.checkBox_LBP.setObjectName("checkBox_LBP")
        self.checkBox_HOG = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_HOG.setGeometry(QtCore.QRect(290, 50, 71, 17))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_HOG.setFont(font)
        self.checkBox_HOG.setObjectName("checkBox_HOG")
        self.checkBox_Moments = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_Moments.setGeometry(QtCore.QRect(290, 80, 81, 16))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_Moments.setFont(font)
        self.checkBox_Moments.setObjectName("checkBox_Moments")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 150, 361, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_requete = QtWidgets.QLabel(self.centralwidget)
        self.label_requete.setGeometry(QtCore.QRect(10, 190, 361, 251))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_requete.setFont(font)
        self.label_requete.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_requete.setText("")
        self.label_requete.setScaledContents(True)
        self.label_requete.setAlignment(QtCore.Qt.AlignCenter)
        self.label_requete.setObjectName("label_requete")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(10, 440, 931, 41))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(380, 10, 551, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(940, 10, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        # self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        # self.gridLayoutWidget.setGeometry(QtCore.QRect(380, 180, 551, 251))
        # self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        # self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        # self.gridLayout.setContentsMargins(0, 0, 0, 0)
        # self.gridLayout.setObjectName("gridLayout")
        # Scroll Area
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(380, 190, 551, 251))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        # Contenu du scroll (l'ancien widget)
        self.gridLayoutWidget = QtWidgets.QWidget()
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")

        # Layout à l'intérieur du widget
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        # Associer le widget au scroll
        self.scrollArea.setWidget(self.gridLayoutWidget)

        self.label_courbe = QtWidgets.QLabel(self.centralwidget)
        self.label_courbe.setGeometry(QtCore.QRect(940, 190, 251, 251))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_courbe.setFont(font)
        self.label_courbe.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_courbe.setText("")
        self.label_courbe.setScaledContents(True)
        self.label_courbe.setAlignment(QtCore.Qt.AlignCenter)
        self.label_courbe.setObjectName("label_courbe")
        self.Quitter = QtWidgets.QPushButton(self.centralwidget)
        self.Quitter.setGeometry(QtCore.QRect(940, 440, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.Quitter.setFont(font)
        self.Quitter.setObjectName("Quitter")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(690, 70, 131, 41))
        self.comboBox.setObjectName("comboBox")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(590, 70, 101, 41))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(380, 150, 551, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(940, 150, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.chercher = QtWidgets.QPushButton(self.centralwidget)
        self.chercher.setGeometry(QtCore.QRect(830, 70, 101, 41))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.chercher.setFont(font)
        self.chercher.setObjectName("chercher")
        self.calcul_RP = QtWidgets.QPushButton(self.centralwidget)
        self.calcul_RP.setGeometry(QtCore.QRect(940, 70, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.calcul_RP.setFont(font)
        self.calcul_RP.setObjectName("calcul_RP")
        self.checkBox_autre = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_autre.setGeometry(QtCore.QRect(290, 110, 81, 17))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_autre.setFont(font)
        self.checkBox_autre.setObjectName("checkBox_autre")
        self.charger = QtWidgets.QPushButton(self.centralwidget)
        self.charger.setGeometry(QtCore.QRect(10, 60, 91, 61))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.charger.setFont(font)
        self.charger.setObjectName("charger")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 10, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.charger_desc = QtWidgets.QPushButton(self.centralwidget)
        self.charger_desc.setGeometry(QtCore.QRect(380, 70, 201, 41))
        # Label pour le nombre d'images
        self.label_nb_images = QtWidgets.QLabel(self.centralwidget)
        self.label_nb_images.setGeometry(QtCore.QRect(380, 120, 251, 22))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_nb_images.setFont(font)
        self.label_nb_images.setAlignment(QtCore.Qt.AlignLeft)
        self.label_nb_images.setObjectName("label_nb_images")

        # Champ pour sélectionner le nombre d'images à afficher
        self.spinBox_nb_images = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_nb_images.setGeometry(QtCore.QRect(640, 118, 71, 22))
        self.spinBox_nb_images.setMinimum(1)
        self.spinBox_nb_images.setMaximum(1000)
        self.spinBox_nb_images.setValue(10)  # valeur par défaut
        self.spinBox_nb_images.setObjectName("spinBox_nb_images")

        self.checkBox_race = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_race.setGeometry(QtCore.QRect(750, 118, 150, 21))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_race.setFont(font)
        self.checkBox_race.setObjectName("checkBox_race")

        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.charger_desc.setFont(font)
        self.charger_desc.setObjectName("charger_desc")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1203, 31))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Choix de descripteur"))
        self.checkBox_HistC.setText(_translate("MainWindow", "BGR"))
        self.checkBox_HSV.setText(_translate("MainWindow", "HSV"))
        self.checkBox_SIFT.setText(_translate("MainWindow", "SIFT"))
        self.checkBox_ORB.setText(_translate("MainWindow", "ORB"))
        self.checkBox_GLCM.setText(_translate("MainWindow", "GLCM"))
        self.checkBox_LBP.setText(_translate("MainWindow", "LBP"))
        self.checkBox_HOG.setText(_translate("MainWindow", "HOG"))
        self.checkBox_Moments.setText(_translate("MainWindow", "Mom."))
        self.label_2.setText(_translate("MainWindow", "Image requête"))
        self.label_4.setText(_translate("MainWindow", "Recherche"))
        self.label_5.setText(_translate("MainWindow", "Rappel/Précision"))
        self.Quitter.setText(_translate("MainWindow", "Quitter"))
        self.label_7.setText(_translate("MainWindow", "Distance :"))
        self.label_8.setText(_translate("MainWindow", "Résultats"))
        self.label_9.setText(_translate("MainWindow", "Courbe R/P"))
        self.chercher.setText(_translate("MainWindow", "Recherche"))
        self.calcul_RP.setText(_translate("MainWindow", "Calculer la courbe R/P"))
        self.checkBox_autre.setText(_translate("MainWindow", "Autre"))
        self.charger.setText(_translate("MainWindow", "Charger"))
        self.label_3.setText(_translate("MainWindow", "Requête"))
        self.charger_desc.setText(_translate("MainWindow", "Charger descripteurs"))
        self.charger.clicked.connect(self.Ouvrir)
        self.Quitter.clicked.connect(self.quitter)
        self.charger_desc.clicked.connect(self.loadFeatures)
        self.chercher.clicked.connect(self.Recherche)
        self.calcul_RP.clicked.connect(self.rappel_precision)
        self.label_nb_images.setText(_translate("MainWindow", "Nombre d’images à afficher :"))
        self.spinBox_nb_images.setValue(10)  # valeur par défaut
        self.spinBox_nb_images.setMinimum(1)  # valeur minimale
        self.checkBox_race.setText(_translate("MainWindow", "Race"))



    def Ouvrir(self, MainWindow): 
        global fileName 
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpeg *.jpg *.bmp)") 
        pixmap = QtGui.QPixmap(fileName) 
        pixmap = pixmap.scaled(self.label_requete.width(), 
        self.label_requete.height(), QtCore.Qt.KeepAspectRatio) 
        self.label_requete.setPixmap(pixmap) 
        self.label_requete.setAlignment(QtCore.Qt.AlignCenter) 
    
    def quitter(self, MainWindow):
        exit()

    def loadFeatures(self, MainWindow): 
        
        folder_model="" 
        if self.checkBox_HistC.isChecked(): 
            folder_model = './BGR' 
            self.algo_choice=1 
        if self.checkBox_HSV.isChecked(): 
            folder_model = './HSV' 
            self.algo_choice=2 
        if self.checkBox_SIFT.isChecked(): 
            folder_model = './SIFT'
            self.algo_choice=3 
        if self.checkBox_ORB.isChecked(): 
            folder_model = './ORB'
            self.algo_choice=4 
        for i in reversed(range(self.gridLayout.count())):
            self.gridLayout.itemAt(i).widget().setParent(None)
        if filenames: 
            if self.algo_choice==3 or self.algo_choice==4: 
                self.comboBox.clear() 
                self.comboBox.addItems(["Brute force","Flann"]) 
            else : 
                self.comboBox.clear() 
                self.comboBox.addItems(["Euclidienne","Correlation","Chi carre","Intersection","Bhattacharyya"]) 
        if len(filenames)<1: 
            print("Merci de charger une image avec le bouton Ouvrir") 
            ##Charger les features de la base de données. 
        n_fichiers = compter_fichiers(folder_model)
        self.features1 = [] 
        pas = 0 
        print("chargement de descripteurs en cours ...") 
        print(os.getcwd())

        for classe in os.listdir(folder_model):
            path_classe = os.path.join(folder_model, classe)
            if not os.path.isdir(path_classe):
                continue

            for sub_class in os.listdir(path_classe):
                path_subclass = os.path.join(path_classe, sub_class)
                if not os.path.isdir(path_subclass):
                    continue

                for fichier in os.listdir(path_subclass):
                    if not fichier.endswith(".npy"):
                        continue

                    chemin_fichier_feature = os.path.join(path_subclass, fichier)
                    feature = np.load(chemin_fichier_feature)

                    # Récupère le nom de l’image originale (remplace .npy par .jpg)
                    nom_image = fichier.split('.')[0] + ".jpg"
                    chemin_image = os.path.join(filenames, classe, sub_class, nom_image)

                    self.features1.append((chemin_image, feature))

                    pas += 1
                    self.progressBar.setValue(int(100 * ((pas + 1) /n_fichiers)))
                    print("chemin_image", chemin_image)
                    #print("feature", feature)
                    

        if not self.checkBox_SIFT.isChecked() and not self.checkBox_HistC.isChecked() and not self.checkBox_HSV.isChecked() and not self.checkBox_ORB.isChecked() : 
            print("Merci de sélectionner au moins un descripteur dans le menu") 
            showDialog()
    def Recherche(self, MainWindow): 
        
        #Remise à 0 de la grille des voisins 
        for i in reversed(range(self.gridLayout.count())): 
            self.gridLayout.itemAt(i).widget().setParent(None) 
            voisins="" 
        if self.algo_choice !=0: 
            ##Generer les features de l'images requete 
            req = extractReqFeatures(fileName, self.algo_choice) 
            ##Definition du nombre de voisins 
            nb_images = self.spinBox_nb_images.value()

            self.sortie =  nb_images
            #Aller chercher dans la liste de l'interface la distance choisie 
            distanceName=self.comboBox.currentText() 
            #Générer les voisins 
            voisins=getkVoisins(self.features1, req, self.sortie, distanceName ) 
            #print(voisins[0][0])
            print(os.path.basename(voisins[0][0]))
            self.path_image_plus_proches = [] 
            self.nom_image_plus_proches = [] 
            for k in range(self.sortie): 
                self.path_image_plus_proches.append(voisins[k][0]) 
                self.nom_image_plus_proches.append(os.path.basename(voisins[k][0])) 
                print(f"nom = {os.path.basename(voisins[k][0])}")
                print(f"path = {voisins[k][0]}")
            #Nombre de colonnes pour l'affichage 
            col=3 
            k=0 
            for i in range(math.ceil(self.sortie/col)): 
                for j in range(col): 
                    print(f"path_image_plus_proches[{k}]",self.path_image_plus_proches[k])
                    img = cv2.imread(self.path_image_plus_proches[k],1) #load image 
                    #Remise de l'image en RGB pour l'afficher correctement 
                    if img is None:
                        print("Erreur lors du chargement de l'image :", self.path_image_plus_proches[k])
                        break
                    b,g,r = cv2.split(img) 
                    img = cv2.merge([r,g,b]) 
                    # get b,g,r 
                    # switch it to rgb 
                    #convert image to QImage 
                    height, width, channel = img.shape 
                    bytesPerLine = 3 * width 
                    qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888) 
                    pixmap=QtGui.QPixmap.fromImage(qImg) 
                    label = QtWidgets.QLabel("") 
                    label.setAlignment(QtCore.Qt.AlignCenter) 
                    label.setPixmap(pixmap.scaled(int(0.3*width), int(0.3*height), 
                    QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation)) 
                    self.gridLayout.addWidget(label, i, j) 
                    k+=1 
        else : 
            print("Il faut choisir une méthode !")


    def rappel_precision(self): 
        folder_model = './MIR_DATASETS_B/MIR_DATASETS_B' 

        self.base_image = glob.glob(os.path.join(folder_model, "**", "*.jpg"), recursive=True)
        race = self.checkBox_race.isChecked()
        rp = []
        


        if race :
            classe_1 =os.path.basename(fileName).split("_")[3]
        else :
            classe_1 =os.path.basename(fileName).split("_")[2]
        for j in range(self.sortie):
            if race:
                classe_2 = os.path.basename(self.path_image_plus_proches[j]).split("_")[3]
            else:
                classe_2 = os.path.basename(self.path_image_plus_proches[j]).split("_")[2]
            
            if classe_1 == classe_2:
                rp.append("pertinent")
            else:
                rp.append("non pertinent")

        val = 0  # Nombre d'images pertinentes accumulées
        precisions = []
        rappels = []
        total_pertinents = 0
        for path in self.base_image:
            basename = os.path.basename(path)
            parts = basename.split("_")
            index = 3 if race else 2
            if len(parts) > index:
                valeur = parts[index]
                if valeur == classe_1:
                    total_pertinents += 1
        total_pertinents = min(total_pertinents, self.sortie)

        for i in range(self.sortie):
            if rp[i] == "pertinent":
                val += 1
            precision = val / (i + 1)
            rappel = val / total_pertinents if total_pertinents != 0 else 0
            precisions.append(precision)
            rappels.append(rappel)

        #Création de la courbe R/P 
        plt.plot(rappels,precisions,linewidth=2) 
        plt.xlabel("Recall",fontsize=12) 
        plt.ylabel("Precision",fontsize=12) 
        plt.title("R/P"+str(self.sortie)+" voisins de l'image " +   os.path.basename(fileName),fontsize=14) 
        #Enregistrement de la courbe RP 
        if race:
            folder_name = os.path.basename(fileName).split("_")[3]
        else:
            folder_name = os.path.basename(fileName).split("_")[2]
        folder_name = "Recall_Precision" + "/" + folder_name 
        save_folder = os.path.join(".", folder_name)
        
        if not os.path.exists(save_folder): 
            os.makedirs(save_folder) 
        save_name=os.path.join(save_folder,os.path.basename(fileName)+'.png') 
        plt.savefig(save_name,format='png',dpi=800) 
        plt.close() 
        #Affichage de la courbe RP
        qImg = QtGui.QImage(save_name)
        pixmap = QtGui.QPixmap.fromImage(qImg)

        width = self.label_requete.frameGeometry().width() 
        height = self.label_requete.frameGeometry().height() 
        self.label_courbe.setPixmap(
            pixmap.scaled(
                self.label_courbe.width(),
                self.label_courbe.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
        )


        # self.label_courbe.setAlignment(QtCore.Qt.AlignCenter) 
        # self.label_courbe.setPixmap(pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))

if __name__ == "__main__":
    import sys
    global filenames
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    
    filenames = os.getcwd() + "/MIR_DATASETS_B/MIR_DATASETS_B" 
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
