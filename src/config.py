# src/config.py
# Centralized paths and constants used across the project.

import os

# Detect environment
if "COLAB_GPU" in os.environ:   # running on Google Colab
    ROOT = "/content/drive/MyDrive/revealai"
else:  # local PC
    ROOT = r"N:\Datasets"   # ✅ project root

DATA_DIR = ROOT  # raw + splits live directly under N:\Datasets

# Raw input folders (put your raw video/audio here)
VIDEO_RAW = os.path.join(DATA_DIR, "video", "raw")
AUDIO_RAW = os.path.join(DATA_DIR, "audio", "raw")

# Preprocessed outputs (created by data_prep.py)
VIDEO_FRAMES = os.path.join(DATA_DIR, "video", "frames")   # frames/real/, frames/fake/
AUDIO_SPECS = os.path.join(DATA_DIR, "audio", "specs")     # specs/real/, specs/fake/

# Splits
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

# Models and reports
MODELS_DIR = os.path.join(ROOT, "models")
VIDEO_MODEL_PATH = os.path.join(MODELS_DIR, "video_xception.h5")
AUDIO_MODEL_PATH = os.path.join(MODELS_DIR, "audio_cnn.h5")
REPORTS_DIR = os.path.join(ROOT, "reports")

# Misc
IMG_SIZE_VIDEO = (299, 299)   # Xception input
IMG_SIZE_AUDIO = (224, 224)   # spectrogram image size for audio CNN
