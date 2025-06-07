# Start from the base image provided by your teacher
FROM coolsa/pyqt-designer


RUN apt update && apt install -y git

# Install necessary Python packages
RUN pip install --no-cache-dir torch torchvision transformers timm pillow sentence-transformers datasets faiss-cpu pandas huggingface_hub tqdm

RUN pip install git+https://github.com/openai/CLIP.git

# Set environment variable for Hugging Face cache directory
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Pre-download the ViT model during the build
RUN python3 -c "from transformers import ViTModel; import timm; timm.create_model('vit_base_patch16_224', pretrained=True)"

RUN python3 -c "from sentence_transformers import SentenceTransformer ; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') "


# RUN python3 -c "from flickr30k import Dataset; import matplotlib.pyplot as plt; from PIL import Image; import clip; import torch; from PIL import Image; import numpy as np; from tqdm import tqdm; import faiss;  Build the dataseta; builder = Dataset(); builder.download_and_prepare();ds = builder.as_dataset();model, preprocess = clip.load('ViT-B/32')"

COPY init_docker.py /opt/TP/init_docker.py
COPY flickr30k.py /opt/TP/flickr30k.py

RUN python3 init_docker.py