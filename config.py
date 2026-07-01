import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Specific files
WIKI_MAT_PATH = os.path.join(RAW_DATA_DIR, "wiki.mat")
WIKI_IMAGE_DIR = os.path.join(RAW_DATA_DIR, "wiki_crop")

PROCESSED_FACE_DIR = os.path.join(PROCESSED_DATA_DIR, "faces")