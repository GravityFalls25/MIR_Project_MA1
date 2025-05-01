import numpy as np
import math
import cv2
#from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from skimage import feature
from matplotlib import pyplot as plt
from skimage.feature import hog,local_binary_pattern
from skimage.feature.texture import graycomatrix, graycoprops
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
    a = np.float32(np.array(a))
    b = np.float32(np.array(b))
    if a.shape[0]==0 or b.shape[0]==0:
        return np.inf
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        print("Descripteurs vides (None ou len == 0)")
        return np.inf

    if len(b.shape) != 2 or len(a.shape) != 2:
        print(f"Mauvais format de descripteurs : a.shape={a.shape}, b.shape={b.shape}")
        return np.inf
    index_params = dict(algorithm=1, trees=5)
    sch_params = dict(checks=50)
    flannMatcher = cv2.FlannBasedMatcher(index_params, sch_params)
    matches = list(map(lambda x: x.distance, flannMatcher.match(a, b)))
    return np.mean(matches)

def bruteForceMatching(a, b):
    a = np.array(a).astype('uint8')
    b = np.array(b).astype('uint8')
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        print("Descripteurs vides (None ou len == 0)")
        return np.inf

    if len(b.shape) != 2 or len(a.shape) != 2:
        print(f"Mauvais format de descripteurs : a.shape={a.shape}, b.shape={b.shape}")
        return np.inf
    matches = list(map(lambda x: x.distance, bf.match(a, b)))
    return np.mean(matches)

def distance_f(l1,l2,distanceName):
    if distanceName=="Euclidienne":
        distance =distance = np.linalg.norm(np.array(l1) - np.array(l2))
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
    for i in range(len(lfeatures)): 
        dist = distance_f(req, lfeatures[i][1],distanceName)
        ldistances.append((lfeatures[i][0], lfeatures[i][1], dist)) 
    if distanceName in ["Correlation","Intersection"]:
        ordre=True
    else:
        ordre=False
    ldistances.sort(key=operator.itemgetter(2),reverse=ordre) 

    lvoisins = [] 
    for i in range(k): 
        lvoisins.append(ldistances[i]) 
    return lvoisins