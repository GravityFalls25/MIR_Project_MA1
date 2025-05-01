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
from skimage.feature.texture import graycomatrix, graycoprops

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
        for sub_class in os.listdir(os.path.join(Dossier_images, classe)):
            path_save = "HSV/" + classe + "/" + sub_class
            if not os.path.isdir(path_save):
                os.mkdir(path_save)
            print(sub_class)
            for path in os.listdir(os.path.join(Dossier_images, classe, sub_class)):
                print(path)
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
                np.savetxt(path_save + "/" + str(num_image) + ".txt", feature)
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
        for sub_class in os.listdir(os.path.join(Dossier_images, classe)):
            path_save = "BGR/" + classe + "/" + sub_class
            if not os.path.isdir(path_save):
                os.mkdir(path_save)
            print(sub_class)
            for path in os.listdir(os.path.join(Dossier_images, classe, sub_class)):
                print(path)
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
                np.savetxt(path_save + "/" + str(num_image) + ".txt", feature)
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
        for sub_class in os.listdir(Dossier_images + "/" + classe):
            path_save = "SIFT/" + classe + "/" + sub_class
            if not os.path.isdir(path_save):
                os.mkdir(path_save)
            print(sub_class)
            for path in os.listdir(Dossier_images + "/" + classe + "/" + sub_class):
                print(path)
                img = cv2.imread(Dossier_images + "/" + classe + "/" + sub_class + "/" + path)
                #img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2) 
                sift = cv2.SIFT_create()
                kps, des = sift.detectAndCompute(img, None)

                if des is None:
                    list_dont.append(Dossier_images + "/" + classe + "/" + sub_class + "/" + path)
                    continue

                num_image, _ = path.split(".")
                np.savetxt(path_save + "/" + str(num_image) + ".txt", des)
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

        for sub_class in os.listdir(Dossier_images +"/"+  classe):
            path_save = "ORB/" + classe + "/" + sub_class 
            if not os.path.isdir(path_save):
                os.mkdir(path_save)
            # sub_class = classe + "/" + sub_class
            print(sub_class)
            for path in os.listdir(Dossier_images +"/" + classe +"/"+ sub_class):
                img = cv2.imread(Dossier_images +"/" + classe +"/"+ sub_class + "/" + path )

                print(path)
                orb = cv2.ORB_create()
                key_point1,descrip1 = orb.detectAndCompute(img,None)
                if descrip1 is None:
                    print(Dossier_images + "/" + classe +"/"+ sub_class + "/" + path + "n'a pas de descripteur "  )
                    list_dont.append(Dossier_images + "/" + classe +"/"+ sub_class + "/" + path)
                    continue
                num_image, _ = path.split(".")
                np.savetxt(path_save + "/" +str(num_image)+".txt" ,descrip1 )
                progressBar.setValue(int(100*((i+1)/n_fichier)))
                i+=1
    print("liste des images qui n'ont pas fonctionnées")
    print(list_dont)
    print(len(list_dont))
    print("indexation ORB terminée !!!!")
	
def extractReqFeatures(fileName,algo_choice):  
    print(algo_choice)
    if fileName : 
        img = cv2.imread(fileName)
        resized_img = resize(img, (128*4, 64*4))
            
        if algo_choice==1: #Couleurs
            histB = cv2.calcHist([img],[0],None,[256],[0,256])
            histG = cv2.calcHist([img],[1],None,[256],[0,256])
            histR = cv2.calcHist([img],[2],None,[256],[0,256])
            vect_features = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)
        
        elif algo_choice==2: # Histo HSV
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([hsv],[0],None,[256],[0,256])
            histS = cv2.calcHist([hsv],[1],None,[256],[0,256])
            histV = cv2.calcHist([hsv],[2],None,[256],[0,256])
            vect_features = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

        elif algo_choice==3: #SIFT
            sift = cv2.SIFT_create() #cv2.xfeatures2d.SIFT_create() pour py < 3.4 
            # Find the key point
            kps , vect_features = sift.detectAndCompute(img,None)
    
        elif algo_choice==4: #ORB
            orb = cv2.ORB_create()
            # finding key points and descriptors of both images using detectAndCompute() function
            key_point1,vect_features = orb.detectAndCompute(img,None)
			
        np.savetxt("Methode_"+str(algo_choice)+"_requete.txt" ,vect_features)
        print("saved")
        #print("vect_features", vect_features)
        return vect_features
    


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

        for sub_class in os.listdir(os.path.join(Dossier_images, classe)):
            print(sub_class)
            path_save = "GLCM/" + classe + "/" + sub_class
            if not os.path.isdir(path_save):
                os.mkdir(path_save)

            for path in os.listdir(os.path.join(Dossier_images, classe, sub_class)):
                print(path)
                full_path = os.path.join(Dossier_images, classe, sub_class, path)
                image = cv2.imread(full_path)
                if image is None:
                    list_dont.append(full_path)
                    continue

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                gray = img_as_ubyte(gray) 

                glcmMatrix = graycomatrix(gray, distances=distances, angles=angles, normed=True) 
                glcmProperties = []
                for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                    glcmProperties.append(graycoprops(glcmMatrix, prop).ravel())
                
                feature = np.concatenate(glcmProperties, axis=None)
                
                num_image, _ = path.split(".") 
                np.savetxt(os.path.join(path_save, str(num_image) + ".txt"), feature) 
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

        for sub_class in os.listdir(os.path.join(Dossier_images, classe)):
            print(sub_class)
            path_save = "LBP/" + classe + "/" + sub_class
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
                np.savetxt(os.path.join(path_save, str(num_image) + ".txt"), histograms) 
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
                np.savetxt(os.path.join(path_save, str(num_image) + ".txt"), feature) 
                progressBar.setValue(int(100 * ((i + 1) / n_fichier))) 
                i += 1 

    print("liste des images qui n'ont pas fonctionnées")
    print(list_dont)
    print(len(list_dont))
    print("indexation HOG terminée !!!!")

