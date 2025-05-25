# Start from the base image provided by your teacher
FROM coolsa/pyqt-designer

# Install necessary Python packages
RUN pip install --no-cache-dir torch torchvision transformers timm pillow sentence-transformers

# Set environment variable for Hugging Face cache directory
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Pre-download the ViT model during the build
RUN python3 -c "from transformers import ViTModel; import timm; timm.create_model('vit_base_patch16_224', pretrained=True)"

RUN python3 -c "from sentence_transformers import SentenceTransformer ; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') "
