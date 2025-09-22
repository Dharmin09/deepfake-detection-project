# src/config.py
# Centralized paths and constants used across the project.

import os

# --- Environment Detection ---
# This block automatically switches paths based on where the code is running.
if "COLAB_GPU" in os.environ:
    # --- Colab Paths ---
    # Path to the main project folder on Google Drive
    PROJECT_ROOT = "/content/drive/MyDrive/revealai"
    # Path to the nested folder containing your datasets
    DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets")
else:
    # --- Local PC Paths ---
    # On your local machine, everything is under one root folder
    PROJECT_ROOT = r"N:\Datasets"
    DATA_ROOT = r"N:\Datasets"

# --- Data Directories (using DATA_ROOT) ---
# These paths point to your audio, video, and splits folders.
VIDEO_RAW = os.path.join(DATA_ROOT, "video", "raw")
AUDIO_RAW = os.path.join(DATA_ROOT, "audio", "raw")
VIDEO_FRAMES = os.path.join(DATA_ROOT, "video", "frames")
AUDIO_SPECS = os.path.join(DATA_ROOT, "audio", "specs")
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")

# --- Asset Directories (using PROJECT_ROOT) ---
# These paths point to where models and reports will be saved.
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# --- Model Paths ---
# These are the final output paths for your trained models.
VIDEO_MODEL_PATH = os.path.join(MODELS_DIR, "video_xception.h5")
AUDIO_MODEL_PATH = os.path.join(MODELS_DIR, "audio_cnn.h5")

# --- Model Configuration ---
# These settings must match the input size expected by the models.
IMG_SIZE_VIDEO = (299, 299)   # Xception input size
IMG_SIZE_AUDIO = (224, 224)   # Spectrogram image size for the audio CNN
