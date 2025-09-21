# src/prepare_video.py
# Extract frames from videos (mp4) → JPGs

import os
import cv2
from pathlib import Path
from tqdm import tqdm  # <--- 1. Import tqdm
from src.config import VIDEO_RAW, VIDEO_FRAMES

os.makedirs(os.path.join(VIDEO_FRAMES, "real"), exist_ok=True)
os.makedirs(os.path.join(VIDEO_FRAMES, "fake"), exist_ok=True)

def extract_frames(in_path, out_dir, every_n=30, max_frames=30):
    # This function remains unchanged
    cap = cv2.VideoCapture(in_path)
    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            out_path = os.path.join(out_dir, f"{Path(in_path).stem}_{saved}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
            if saved >= max_frames:
                break
        count += 1
    cap.release()

def process_videos(label):
    in_dir = os.path.join(VIDEO_RAW, label)
    out_dir = os.path.join(VIDEO_FRAMES, label)
    
    # --- 2. Get a list of files to process ---
    video_files = [f for f in os.listdir(in_dir) if f.endswith(".mp4")]
    
    # --- 3. Wrap the file list with tqdm for a progress bar ---
    for fname in tqdm(video_files, desc=f"Processing {label} videos"):
        in_path = os.path.join(in_dir, fname)
        extract_frames(in_path, out_dir)

if __name__ == "__main__":
    for lbl in ["real", "fake"]:
        # The old print statement was removed as tqdm handles the description now
        process_videos(lbl)
    print("\n✅ All frames saved in:", VIDEO_FRAMES)