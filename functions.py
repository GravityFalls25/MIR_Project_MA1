#Defintion de toute les fonctions à appeller dans l'interface
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import os
import cv2
import numpy as np
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage import io, color, img_as_ubyte
from matplotlib import pyplot as plt
from skimage.feature import hog, local_binary_pattern
from skimage.feature.texture import greycomatrix, greycoprops
import torchvision.transforms as transforms
import torch
import timm
from tqdm import tqdm
from PIL import Image

def showDialog():
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Information)
    msgBox.setText("Merci de sélectionner un descripteur via le menu ci-dessus")
    msgBox.setWindowTitle("Pas de Descripteur sélectionné")
    msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    returnValue = msgBox.exec()

def generateHistogramme_HSV(Dossier_images, progressBar):
    list_dont = []
    n_fichier = compter_fichiers(Dossier_images)
    if not os.path.isdir("HSV"):
        os.mkdir("HSV")
    i = 0
    path_save = ""
    print(Dossier_images)
    for classe in os.listdir(Dossier_images):
        path_save = "HSV/" + classe
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        print(classe)
        for sub_class in tqdm(os.listdir(os.path.join(Dossier_images, classe))):
            path_save = "HSV/" + classe + "/" + sub_class
            if not os.path.isdir(path_save):
                os.mkdir(path_save)
            # print(sub_class)
            for path in os.listdir(os.path.join(Dossier_images, classe, sub_class)):
                # print(path)
                full_path = os.path.join(Dossier_images, classe, sub_class, path)
                img = cv2.imread(full_path)
                if img is None:
                    list_dont.append(full_path)
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                histH = cv2.calcHist([img], [0], None, [256], [0, 256])
                histS = cv2.calcHist([img], [1], None, [256], [0, 256])
                histV = cv2.calcHist([img], [2], None, [256], [0, 256])
                feature = np.concatenate((histH, histS, histV), axis=None)

                num_image, _ = path.split(".")
                np.save(path_save + "/" + str(num_image) + ".txt", feature)
                if progressBar is not None:
                    progressBar.setValue(int(100 * ((i + 1) / n_fichier)))
                i += 1

    print("liste des images qui n'ont pas fonctionnées")
    print(list_dont)
    print(len(list_dont))
    print("indexation Hist HSV terminée !!!!")

def generateHistogramme_Color(Dossier_images, progressBar):
    list_dont = []
    n_fichier = compter_fichiers(Dossier_images)
    if not os.path.isdir("BGR"):
        os.mkdir("BGR")
    i = 0
    path_save = ""
    print(Dossier_images)
    for classe in os.listdir(Dossier_images):
        path_save = "BGR/" + classe
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        print(classe)
        for sub_class in tqdm(os.listdir(os.path.join(Dossier_images, classe))):
            path_save = "BGR/" + classe + "/" + sub_class
            if not os.path.isdir(path_save):
                os.mkdir(path_save)
            # print(sub_class)
            for path in os.listdir(os.path.join(Dossier_images, classe, sub_class)):
                # print(path)
                full_path = os.path.join(Dossier_images, classe, sub_class, path)
                img = cv2.imread(full_path)
                if img is None:
                    list_dont.append(full_path)
                    continue

                histB = cv2.calcHist([img], [0], None, [256], [0, 256])
                histG = cv2.calcHist([img], [1], None, [256], [0, 256])
                histR = cv2.calcHist([img], [2], None, [256], [0, 256])
                feature = np.concatenate((histB, histG, histR), axis=None)

                num_image, _ = path.split(".")
                np.save(path_save + "/" + str(num_image) + ".txt", feature)
                if progressBar is not None:
                    progressBar.setValue(int(100 * ((i + 1) / n_fichier)))
                i += 1

    print("liste des images qui n'ont pas fonctionnées")
    print(list_dont)
    print(len(list_dont))
    print("indexation Hist Couleur terminée !!!!")

def generateSIFT(Dossier_images, progressBar):
    list_dont = []
    n_fichier = compter_fichiers(Dossier_images)
    if not os.path.isdir("SIFT"):
        os.mkdir("SIFT")
    i = 0
    path_save = ""
    print(Dossier_images)
    for classe in os.listdir(Dossier_images):
        path_save = "SIFT/" + classe
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        print(classe)
        for sub_class in tqdm(os.listdir(Dossier_images + "/" + classe)):
            path_save = "SIFT/" + classe + "/" + sub_class
            if not os.path.isdir(path_save):
                os.mkdir(path_save)
            # print(sub_class)
            for path in (os.listdir(Dossier_images + "/" + classe + "/" + sub_class)):
                # print(path)
                img = cv2.imread(Dossier_images + "/" + classe + "/" + sub_class + "/" + path)
                #img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2) 
                sift = cv2.SIFT_create()
                kps, des = sift.detectAndCompute(img, None)

                if des is None:
                    list_dont.append(Dossier_images + "/" + classe + "/" + sub_class + "/" + path)
                    continue

                num_image, _ = path.split(".")
                np.save(path_save + "/" + str(num_image) + ".txt", des)
                if progressBar is not None:
                    progressBar.setValue(int(100 * ((i + 1) / n_fichier)))
                i += 1

    print("liste des images qui n'ont pas fonctionnées")
    print(list_dont)
    print(len(list_dont))
    print("indexation SIFT terminée !!!!")    

def compter_fichiers(dossier):
    compteur = 0
    for racine, sous_dossiers, fichiers in os.walk(dossier):
        compteur += len(fichiers)
    return compteur 

def generateORB(Dossier_images, progressBar):
    list_dont = []
    n_fichier = compter_fichiers(Dossier_images)
    if not os.path.isdir("ORB"):
        os.mkdir("ORB")
    i=0
    path_save = ""
    print(Dossier_images)
    for classe in os.listdir(Dossier_images):
        path_save = "ORB/" + classe
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        # classe = Dossier_images  + "/" + classe
        print(classe)

        for sub_class in tqdm(os.listdir(Dossier_images +"/"+  classe)):
            path_save = "ORB/" + classe + "/" + sub_class 
            if not os.path.isdir(path_save):
                os.mkdir(path_save)
            # sub_class = classe + "/" + sub_class
            # print(sub_class)
            for path in (os.listdir(Dossier_images +"/" + classe +"/"+ sub_class)):
                img = cv2.imread(Dossier_images +"/" + classe +"/"+ sub_class + "/" + path )

                # print(path)
                orb = cv2.ORB_create()
                key_point1,descrip1 = orb.detectAndCompute(img,None)
                if descrip1 is None:
                    print(Dossier_images + "/" + classe +"/"+ sub_class + "/" + path + "n'a pas de descripteur "  )
                    list_dont.append(Dossier_images + "/" + classe +"/"+ sub_class + "/" + path)
                    continue
                num_image, _ = path.split(".")
                np.save(path_save + "/" +str(num_image)+".txt" ,descrip1 )
                if progressBar is not None:
                    progressBar.setValue(int(100*((i+1)/n_fichier)))
                i+=1
    print("liste des images qui n'ont pas fonctionnées")
    print(list_dont)
    print(len(list_dont))
    print("indexation ORB terminée !!!!")

def generateViT(Dossier_images,progressBar, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Utilisation de l'appareil :", device)
    feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=True)
    feature_extractor.reset_classifier(0)
    feature_extractor.eval()
    feature_extractor.to(device)

    if not os.path.isdir("Vit_descriptor"):
        os.mkdir("Vit_descriptor")

    list_dont = []
    n_fichier = compter_fichiers(Dossier_images)
    i = 0

    for classe in os.listdir(Dossier_images):
        path_save_classe = os.path.join("ViT", classe)
        if not os.path.isdir(path_save_classe):
            os.mkdir(path_save_classe)

        for sub_class in tqdm(os.listdir(os.path.join(Dossier_images, classe))):
            path_save_sub_class = os.path.join(path_save_classe, sub_class)
            if not os.path.isdir(path_save_sub_class):
                os.mkdir(path_save_sub_class)

            for path in os.listdir(os.path.join(Dossier_images, classe, sub_class)):
                img_path = os.path.join(Dossier_images, classe, sub_class, path)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Image non lisible : {img_path}")
                    list_dont.append(img_path)
                    continue

                img_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = cv2.resize(img_tensor, (224, 224))
                img_tensor = img_tensor.transpose((2, 0, 1))  # Convert to CxHxW
                img_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0) / 255.0
                img_tensor = img_tensor.to(device)

                with torch.no_grad():
                    features = feature_extractor(img_tensor)

                features = features.cpu().numpy().flatten()
                num_image, _ = os.path.splitext(path)
                np.save(os.path.join(path_save_sub_class, f"{num_image}.npy"), features)

                if progressBar is not None:
                    progressBar.setValue(int(100 * ((i + 1) / n_fichier)))
                i += 1

    print("Images sans features ViT:")
    print(list_dont)
    print(len(list_dont))
    print("Indexation ViT terminée !!!!")


def extractReqFeatures(fileName, algo_choice, mode='concat'):
    print("Méthodes sélectionnées :", algo_choice)
    if not fileName:
        return None

    img = cv2.imread(fileName)
    if img is None:
        print("Erreur : image non chargée")
        return None

    features_dict = {}
    features_dict[fileName] = []
    print("algo_choice", algo_choice)
    for algo in algo_choice:
        if algo == "BGR":
            histB = cv2.calcHist([img], [0], None, [256], [0, 256])
            histG = cv2.calcHist([img], [1], None, [256], [0, 256])
            histR = cv2.calcHist([img], [2], None, [256], [0, 256])
            vect_features = np.concatenate((histB, histG, histR), axis=0)
            features_dict[fileName].append(vect_features)

        elif algo == "HSV":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([hsv], [0], None, [256], [0, 256])
            histS = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            histV = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            vect_features = np.concatenate((histH, histS, histV), axis=0)
            features_dict[fileName].append(vect_features)

        elif algo == "SIFT":
            sift = cv2.SIFT_create()
            _, descriptors = sift.detectAndCompute(img, None)
            if descriptors is None:
                descriptors = np.zeros((1, 128))
            vect_features = descriptors
            features_dict[fileName].append(vect_features)
            print("Taille du vecteur SIFT :", vect_features.shape)



 
        elif algo == "ORB":
            orb = cv2.ORB_create()
            _, descriptors = orb.detectAndCompute(img, None)
            if descriptors is None:
                descriptors = np.zeros((1, 32))
            vect_features = descriptors
            features_dict[fileName].append(vect_features)
            print("Taille du vecteur ORB :", vect_features.shape)
        
        elif algo == "Vit":
            # Charge le modèle Vit
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=True)
            feature_extractor.reset_classifier(0)  # Remove classification head to get pure features
            feature_extractor.eval()
            feature_extractor.to(device)
            # Prétraitement de l'image
            img_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = cv2.resize(img_tensor, (224, 224))
            img_tensor = img_tensor.transpose((2, 0, 1))  # Convert to CxHxW
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            img_tensor = img_tensor.to(device) / 255.0  # Normalisation
            # Extraction des caractéristiques
            with torch.no_grad():
                features = feature_extractor(img_tensor)
            features = features.cpu().numpy().flatten()
            features_dict[fileName].append(features)

        else:
            print(f"Descripteur inconnu : {algo}")
            continue

    # print("Vecteurs de caractéristiques extraits pour chaque méthode :", features_dict)
    print("Taille des vecteurs de caractéristiques :", [v.shape for v in features_dict[fileName]])
        
        
    # Fusion via la fonction fournie
    if len(algo_choice) == 1:
        # Si une seule méthode est choisie, on ne fusionne pas
        fused_features = [(fileName, features_dict[fileName][0])]
    else:
        fused_features = fusion_features_dict(features_dict, mode=mode)
    
    print("Taille du vecteur:", fused_features[0][1].shape)

    # Optionnel : sauvegarde
    np.save(f"Requete_fusion_{'_'.join(algo_choice)}_{mode}.npy", fused_features[0][1])
    print("Descripteur fusionné sauvegardé.")

    return fused_features[0][1]  # on retourne juste le vecteur final


    
def fusion_features_dict(features_dict, mode='concat'):
    fused_features = []
    for chemin_image, vecteurs in features_dict.items():
        if mode == 'concat':
            liste = []
            for i,v in enumerate(vecteurs):
                if i == 0:
                    liste = v.copy()
                else:
                    liste = np.hstack((liste, v))

            fused_features.append((chemin_image, liste))
            # vecteur_fusion = np.concatenate(vecteurs)
            # print(vecteur_fusion.shape)
        elif mode == 'moyenne':
            vecteur_fusion = np.mean(np.array(vecteurs), axis=0)
        else:
            raise ValueError(f"Mode de fusion inconnu: {mode}")

        # fused_features.append((chemin_image, vecteur_fusion))
    return fused_features


def generateGLCM(Dossier_images, progressBar): 
    list_dont = []
    if not os.path.isdir("GLCM"): 
        os.mkdir("GLCM") 
    distances = [1, -1] 
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] 
    n_fichier = compter_fichiers(Dossier_images)
    i = 0 
    print(Dossier_images)
    for classe in os.listdir(Dossier_images):
        print(classe)
        path_save = "GLCM/" + classe
        if not os.path.isdir(path_save):
            os.mkdir(path_save)

        for sub_class in tqdm(os.listdir(os.path.join(Dossier_images, classe))):
            # print(sub_class)
            path_save = "GLCM/" + classe + "/" + sub_class
            if not os.path.isdir(path_save):
                os.mkdir(path_save)

            for path in os.listdir(os.path.join(Dossier_images, classe, sub_class)):
                # print(path)
                full_path = os.path.join(Dossier_images, classe, sub_class, path)
                image = cv2.imread(full_path)
                if image is None:
                    list_dont.append(full_path)
                    continue

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                gray = img_as_ubyte(gray) 

                glcmMatrix = greycomatrix(gray, distances=distances, angles=angles, normed=True) 
                glcmProperties = []
                for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                    glcmProperties.append(greycoprops(glcmMatrix, prop).ravel())
                
                feature = np.concatenate(glcmProperties, axis=None)
                
                num_image, _ = path.split(".") 
                np.save(os.path.join(path_save, str(num_image) + ".txt"), feature) 
                if progressBar is not None:
                    progressBar.setValue(int(100 * ((i + 1) / n_fichier)))
                i += 1 

    print("liste des images qui n'ont pas fonctionnées")
    print(list_dont)
    print(len(list_dont))
    print("indexation GLCM terminée !!!!") 

from skimage.feature import local_binary_pattern

def generateLBP(Dossier_images, progressBar): 
    list_dont = []
    if not os.path.isdir("LBP"): 
        os.mkdir("LBP") 
    points = 8 
    radius = 1 
    method = 'default' 
    subSize = (70, 70) 
    n_fichier = compter_fichiers(Dossier_images)
    i = 0 
    print(Dossier_images)
    for classe in os.listdir(Dossier_images):
        print(classe)
        path_save = "LBP/" + classe
        if not os.path.isdir(path_save):
            os.mkdir(path_save)

        for sub_class in tqdm(os.listdir(os.path.join(Dossier_images, classe))):
            # print(sub_class)
            path_save = "LBP/" + classe + "/" + sub_class
            if not os.path.isdir(path_save):
                os.mkdir(path_save)

            for path in os.listdir(os.path.join(Dossier_images, classe, sub_class)):
                # print(path)
                full_path = os.path.join(Dossier_images, classe, sub_class, path)
                img = cv2.imread(full_path)
                if img is None:
                    list_dont.append(full_path)
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                img = cv2.resize(img, (350, 350)) 
                fullLBPmatrix = local_binary_pattern(img, points, radius, method) 

                histograms = [] 
                for k in range(int(fullLBPmatrix.shape[0] / subSize[0])): 
                    for j in range(int(fullLBPmatrix.shape[1] / subSize[1])): 
                        subVector = fullLBPmatrix[
                            k * subSize[0]:(k + 1) * subSize[0],
                            j * subSize[1]:(j + 1) * subSize[1]
                        ].ravel() 
                        subHist, _ = np.histogram(subVector, bins=int(2**points), range=(0, 2**points)) 
                        histograms = np.concatenate((histograms, subHist), axis=None) 

                num_image, _ = path.split(".") 
                np.save(os.path.join(path_save, str(num_image) + ".txt"), histograms) 
                if progressBar is not None:
                    progressBar.setValue(int(100 * ((i + 1) / n_fichier))) 
                i += 1 

    print("liste des images qui n'ont pas fonctionnées")
    print(list_dont)
    print(len(list_dont))
    print("indexation LBP terminée !!!!")

def generateHOG(Dossier_images, progressBar): 
    list_dont = []
    if not os.path.isdir("HOG"): 
        os.mkdir("HOG") 
    n_fichier = compter_fichiers(Dossier_images)
    i = 0 
    print(Dossier_images)
    for classe in os.listdir(Dossier_images):
        print(classe)
        path_save = "HOG/" + classe
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        
        for sub_class in os.listdir(os.path.join(Dossier_images, classe)):
            print(sub_class)
            path_save = "HOG/" + classe + "/" + sub_class
            if not os.path.isdir(path_save):
                os.mkdir(path_save)

            for path in os.listdir(os.path.join(Dossier_images, classe, sub_class)):
                print(path)
                full_path = os.path.join(Dossier_images, classe, sub_class, path)
                img = cv2.imread(full_path)
                if img is None:
                    list_dont.append(full_path)
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (128, 128))  # Taille raisonnable pour HOG

                feature, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), block_norm='L2-Hys',
                                 visualize=True)

                num_image, _ = path.split(".") 
                np.save(os.path.join(path_save, str(num_image) + ".txt"), feature) 
                if progressBar is not None:
                    progressBar.setValue(int(100 * ((i + 1) / n_fichier))) 
                i += 1 

    print("liste des images qui n'ont pas fonctionnées")
    print(list_dont)
    print(len(list_dont))
    print("indexation HOG terminée !!!!")

