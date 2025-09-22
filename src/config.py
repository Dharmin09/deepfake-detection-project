# src/config.py
# Centralized paths and constants used across the project.

import os

# --- Environment Detection ---
# This block automatically switches paths based on where the code is running.
if "COLAB_GPU" in os.environ:
    # This path is used when running on Google Colab.
    # CORRECTED to match your Google Drive screenshot.
    ROOT = "/content/drive/MyDrive/revealai/datasets" 
else:
    # This path is used when running on your local computer.
    ROOT = r"N:\Datasets"

# --- Main Data Directories ---
# These folders contain your raw data, processed data, and results.
VIDEO_RAW = os.path.join(ROOT, "video", "raw")
AUDIO_RAW = os.path.join(ROOT, "audio", "raw")
VIDEO_FRAMES = os.path.join(ROOT, "video", "frames")
AUDIO_SPECS = os.path.join(ROOT, "audio", "specs")
SPLITS_DIR = os.path.join(ROOT, "splits")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

# --- Model Paths ---
# These are the final output paths for your trained models.
VIDEO_MODEL_PATH = os.path.join(MODELS_DIR, "video_xception.h5")
AUDIO_MODEL_PATH = os.path.join(MODELS_DIR, "audio_cnn.h5")

# --- Model Configuration ---
# These settings must match the input size expected by the models.
IMG_SIZE_VIDEO = (299, 299)   # Xception input size
IMG_SIZE_AUDIO = (224, 224)   # Spectrogram image size for the audio CNN