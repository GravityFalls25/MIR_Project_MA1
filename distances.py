import numpy as np
import math
import cv2
#from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from skimage import feature
from matplotlib import pyplot as plt
from skimage.feature import hog,local_binary_pattern
# from skimage.feature.texture import graycomatrix, graycoprops
from tqdm import tqdm

import operator
import collections 
from collections import Counter

def euclidean(l1, l2):
    pass
    #????????????????????????????????????????

def chiSquareDistance(l1, l2):
    s = 0.0
    for i,j in zip(l1,l2):
        if i == j == 0.0:
            continue
        s += (i - j)**2 / (i + j)
    return s

def bhatta(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    num = np.sum(np.sqrt(np.multiply(l1,l2,dtype=np.float64)),dtype=np.float64)
    den = np.sqrt(np.sum(l1,dtype=np.float64)*np.sum(l2,dtype=np.float64))
    return math.sqrt( 1 - num / den )


def flann(a,b):
    # if a is None or b is None or len(a) == 0 or len(b) == 0:
    #     return np.inf

    # a = np.array(a, dtype=np.float32)
    # b = np.array(b, dtype=np.float32)

    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)

    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.match(a, b)

    # distances = [m.distance for m in matches]

    # return np.mean(distances)
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return np.inf



    a = np.asarray(a)
    b = np.asarray(b)

    if a.dtype == np.uint8:

        
        # ORB / binary descriptors
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=12,
                            key_size=20,
                            multi_probe_level=2)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

    elif a.dtype == np.float32:
        # SIFT / float descriptors
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

    else:
        raise ValueError("Unsupported descriptor type: must be uint8 (ORB) or float32 (SIFT)")

    matches = flann.match(a, b)
    distances = [m.distance for m in matches]

    return np.mean(distances)
    
def bruteForceMatching(a, b):
    a = np.array(a).astype('uint8')
    b = np.array(b).astype('uint8')
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        print("Descripteurs vides (None ou len == 0)")
        return np.inf

    if b.shape != a.shape:
        print(f"Mauvais format de descripteurs : a.shape={a.shape}, b.shape={b.shape}")
        return np.inf
    print(a.shape, b.shape)
    matches = list(map(lambda x: x.distance, bf.match(a, b)))
    return np.mean(matches)

def distance_f(l1,l2,distanceName):
    if distanceName=="Euclidienne":
        distance = np.linalg.norm(np.array(l1) - np.array(l2))
    elif distanceName in ["Correlation","Chi carre","Intersection","Bhattacharyya"]:
        l1 = np.float32(l1)
        l2 = np.float32(l2)
        if distanceName=="Correlation":
            methode=cv2.HISTCMP_CORREL
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName=="Chi carre":
            distance = cv2.compareHist(l1, l2, cv2.HISTCMP_CHISQR)
        elif distanceName=="Intersection":
            methode=cv2.HISTCMP_INTERSECT
            distance = cv2.compareHist(l1, l2, cv2.HISTCMP_INTERSECT)
        elif distanceName=="Bhattacharyya":
            distance = cv2.compareHist(l1, l2, cv2.HISTCMP_BHATTACHARYYA)    
    elif distanceName=="Brute force":
        distance = bruteForceMatching(l1, l2)
    elif distanceName=="Flann":
        distance= flann(l1, l2)
    else:
        distance = bruteForceMatching(l1, l2)
    return distance

def getkVoisins(lfeatures, req, k,distanceName) : 
    ldistances = []
    tmp= 0 
    for i in tqdm(range(len(lfeatures))): 
        # print(req.shape, lfeatures[i][1].flatten().shape)
        dist = distance_f(req.flatten(), lfeatures[i][1].flatten(), distanceName)
        if dist != np.inf:
            tmp +=1
        ldistances.append((lfeatures[i][0], lfeatures[i][1], dist)) 
    if distanceName in ["Correlation","Intersection"]:
        ordre=True
    else:
        ordre=False

    print(f"Nombre de descripteurs valides : {tmp} ")
    
    ldistances.sort(key=operator.itemgetter(2),reverse=ordre) 

    lvoisins = [] 
    for i in range(k): 
        lvoisins.append(ldistances[i]) 
    return lvoisins