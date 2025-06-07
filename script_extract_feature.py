import os

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"utilise: {device}")


from skimage.transform import resize
from skimage.io import imread
from skimage.feature import hog
from skimage import exposure
from matplotlib import pyplot as plt
import functions as fct
import time
import requests
import zipfile
import os
import shutil

from sentence_transformers import SentenceTransformer
import json
import numpy as np

from tqdm import tqdm


import clip

from PIL import Image
import faiss

from flickr30k import Dataset



def check_and_download_dataset_part12():
    if os.path.exists("MIR_DATASETS_B"):
        print("Le dossier existe déjà.")
    else:
        # URL du fichier zip
        url = "https://github.com/sidimahmoudi/facenet_tf2/releases/download/AI_MIR_CLOUD/MIR_DATASETS_B.zip"

        # Nom du fichier zip local
        fichier_zip = "MIR_DATASETS_B.zip"

        # Dossier d'extraction
        dossier_extraction = "MIR_DATASETS_B"

        # 1. Supprimer l'ancien dossier s'il existe
        if os.path.exists(dossier_extraction):
            print(f"Suppression de l'ancien dossier : {dossier_extraction}")
            shutil.rmtree(dossier_extraction)

        # 2. Télécharger le fichier zip
        print("Téléchargement en cours...")
        response = requests.get(url, stream=True)
        with open(fichier_zip, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print("Téléchargement terminé.")

        # 3. Extraire le fichier zip
        print("Extraction en cours...")
        with zipfile.ZipFile(fichier_zip, 'r') as zip_ref:
            zip_ref.extractall(dossier_extraction)
        print(f"Extraction terminée dans le dossier : {dossier_extraction}")

        # 4. (Optionnel) Supprimer le fichier zip
        os.remove(fichier_zip)
        print("Fichier zip supprimé.")
        

        # Chemin vers le dossier à supprimer
        dossier_a_supprimer = 'MIR_DATASETS_B/MIR_DATASETS_B/araignees'

        # Vérifie si le dossier existe avant de le supprimer
        if os.path.exists(dossier_a_supprimer):
            shutil.rmtree(dossier_a_supprimer)
            print("Dossier supprimé avec succès.")
        else:
            print("Le dossier n'existe pas.")


def check_and_download_dataset_part3():
    if os.path.exists("flickr30k-images"):
        print("Le dossier existe déjà.")
    else:
        # URL du fichier zip
        url = "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip"

        # Nom du fichier zip local
        fichier_zip = "flickr30k-images.zip"

        # Dossier d'extraction
        dossier_extraction = "flickr30k-images"


        # 2. Télécharger le fichier zip
        print("Téléchargement en cours...")
        response = requests.get(url, stream=True)
        with open(fichier_zip, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print("Téléchargement terminé.")

        # 3. Extraire le fichier zip
        print("Extraction en cours...")
        with zipfile.ZipFile(fichier_zip, 'r') as zip_ref:
            zip_ref.extractall(dossier_extraction)
        print(f"Extraction terminée dans le dossier : {dossier_extraction}")

        # 4. (Optionnel) Supprimer le fichier zip
        os.remove(fichier_zip)
        print("Fichier zip supprimé.")


def extract_text_features():

    # Open and load the JSON file
    with open('captions_MIR_DATASETS_B.json', 'r') as f:
        text_dict = json.load(f)

    # Remove all keys containing "araignees" from the dictionary
    keys_to_remove = [key for key in text_dict if "araignees" in key]
    for key in keys_to_remove:
        text_dict.pop(key)


    model_text = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') #import le modele

    text_features_dict = {} #va créer un dico avec les embeding

    for image_name, description in tqdm(text_dict.items()): #repasse pour chaque elem du dico text
        embedding = model_text.encode(description) #encode juste le texte
        text_features_dict[image_name] = embedding  #ajoute dane le nouveau dico


    text_descriptor_folder = "text_descriptor"
    os.makedirs(text_descriptor_folder, exist_ok=True)

    for key, embedding in text_features_dict.items():
        # Create subfolders according to the key path
        subfolder = os.path.join(text_descriptor_folder, os.path.dirname(key))
        os.makedirs(subfolder, exist_ok=True)
        filename = os.path.basename(key).split(".")[0] + ".npy"
        filepath = os.path.join(subfolder, filename)
        np.save(filepath, embedding)


def extract_features_flickr30k():
    # Build the dataseta
    builder = Dataset()
    builder.download_and_prepare()

    # Load the dataset
    ds = builder.as_dataset()

    dataset = ds["test"]

    # Check example
    # print(ds["test"][0])
    print(dataset)

    max_char_len = 200
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_embeddings = []
    text_embeddings = []
    image_paths = []
    captions = []

    for sample in tqdm(dataset):
        # print(sample)
        image_path = "flickr30k-images/"+sample['filename']
        caption_list = sample['caption']  # this is a list — pick one, or average later
        

        # Load and encode image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(image)
        image_embedding = image_embedding.cpu().numpy()

        # embed le text
        caption_embeddings = []
        for caption in caption_list:
            text_token = clip.tokenize([caption[:max_char_len]]).to(device)
            with torch.no_grad():
                text_embedding = model.encode_text(text_token)
            text_embedding = text_embedding.cpu().numpy()
            caption_embeddings.append(text_embedding)

        # Average caption embeddings
        caption_embeddings = np.vstack(caption_embeddings)
        mean_caption_embedding = np.mean(caption_embeddings, axis=0, keepdims=True)

        image_embeddings.append(image_embedding)
        text_embeddings.append(mean_caption_embedding)
        image_paths.append(image_path)
        captions.append(caption_list)

    image_embeddings = np.vstack(image_embeddings)
    text_embeddings = np.vstack(text_embeddings)

    #concatene pour faire le mutlimodal
    image_embeddings_norm = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embeddings_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    multimodal_embeddings = np.hstack([image_embeddings_norm, text_embeddings_norm])

    # Save multimodal embeddings if needed
    np.save("flickr30k_clip_multimodal_embeddings.npy", multimodal_embeddings)


    #créate FAISS pour tout indexer
    dimension = image_embeddings.shape[1]
    image_index = faiss.IndexFlatL2(dimension)
    image_index.add(image_embeddings.astype('float32'))

    #stock tout
    faiss.write_index(image_index, "flickr30k_clip_images.index")
    np.save("image_paths.npy", image_paths)

    #pariel pour text
    text_index = faiss.IndexFlatL2(dimension)
    text_index.add(text_embeddings.astype('float32'))

    # Save text index
    faiss.write_index(text_index, "flickr30k_clip_texts.index")
    np.save("captions.npy", captions)


    # Save multimodal index
    multimodal_index = faiss.IndexFlatL2(multimodal_embeddings.shape[1])
    multimodal_index.add(multimodal_embeddings.astype('float32'))
    faiss.write_index(multimodal_index, "flickr30k_clip_multimodal.index")


while True:
    print("Extraction des features")
    print("1) pour la partie 1")
    print("2) pour la partie 2 et 3 (attention cela peut prendre du temps sans GPU)")
    print("3) pour quitter")

    choix = input("Votre choix : ")

    dossier_dataset_1 = os.getcwd() + "/MIR_DATASETS_B/MIR_DATASETS_B"

    if choix == "1":
        print("Indexation de tous les descripteurs utile à la partie 1")
        check_and_download_dataset_part12()

        #histogramme de couleur BGR
        print("start BGR")
        a = time.time()
        fct.generateHistogramme_Color(dossier_dataset_1, None)
        print(f"Histogramme de couleur BGR généré en {time.time()-a} secondes")

        #Histogramme de couleur HSV
        print("start HSV")
        a = time.time()
        fct.generateHistogramme_HSV(dossier_dataset_1, None)
        print(f"Histogramme de couleur HSV généré en {time.time()-a} secondes")

        #SIFT
        print("start SIFT")
        a = time.time()
        fct.generateSIFT(dossier_dataset_1, None)
        print(f"Descripteur SIFT généré en {time.time()-a} secondes")

        #ORB
        print("start ORB")
        a = time.time()
        fct.generateORB(dossier_dataset_1, None)
        print(f"Descripteur ORB généré en {time.time()-a} secondes")

        #GLCM
        print("start GLCM")
        a = time.time()
        fct.generateGLCM(dossier_dataset_1, None)
        print(f"Descripteur GLCM généré en {time.time()-a} secondes")

        #LBP
        print("start LBP")
        a = time.time()
        fct.generateLBP(dossier_dataset_1, None)
        print(f"Descripteur LBP généré en {time.time()-a} secondes")

        #VIT
        print("start VIT")
        a = time.time()
        fct.generateViT(dossier_dataset_1, None, device=device)
        print(f"Descripteur HOG généré en {time.time()-a} secondes")


    elif choix == "2":
        print("indexation que du descripteur texte, partie image déjà utilisé dans la 1er ")
        a = time.time()
        extract_text_features()
        print(f"Descripteur texte généré en {time.time()-a} secondes")

        check_and_download_dataset_part3()

        print("partie 3 avec clip TRES LONG")
        a = time.time()
        extract_features_flickr30k()
        print(f"Descripteur CLIP généré en {time.time()-a} secondes")



    elif choix == "3":
        break
    else:
        print("Choix invalide")