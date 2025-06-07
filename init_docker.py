# prepare_data.py
from flickr30k import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import clip
import torch
import numpy as np
from tqdm import tqdm
import faiss

print("Downloading Flickr30k...")
builder = Dataset()
builder.download_and_prepare()
ds = builder.as_dataset()

print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("Done.")
