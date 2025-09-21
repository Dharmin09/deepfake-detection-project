# utils.py
# Helper functions: frame extraction, spectrogram conversion, Grad-CAM utilities, small I/O helpers.
# Compatible with TF 2.x (tested with TF 2.19 on Colab).

import os
import cv2
import numpy as np
import librosa
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
import io
import matplotlib.cm as cm

# --------------------------
# Video: frame extraction
# --------------------------
def extract_frames(video_path, out_dir, every_n_frames=10, max_frames=50, resize=(299, 299)):
    """
    Extract frames from a video and save as JPEGs into out_dir.
    - every_n_frames: sample every Nth frame (reduces total frames)
    - max_frames: stop after saving this many frames
    - resize: target size (W,H) for saved frames
    Returns number of frames saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n_frames == 0:
            # convert BGR->RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, resize)
            fname = f"{os.path.splitext(os.path.basename(video_path))[0]}_f{saved}.jpg"
            out_path = os.path.join(out_dir, fname)
            Image.fromarray(frame_resized).save(out_path, format="JPEG")
            saved += 1
            if saved >= max_frames:
                break
        idx += 1
    cap.release()
    return saved

def load_image_paths(folder, exts=(".jpg", ".png")):
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if f.lower().endswith(exts)]
    return files

def load_images_to_array(file_list, target_size=(299,299)):
    """
    Returns numpy array shape (N,H,W,3) dtype float32 (not preprocessed)
    """
    arr = []
    for f in file_list:
        img = Image.open(f).convert("RGB").resize(target_size)
        arr.append(np.array(img))
    if len(arr) == 0:
        return np.zeros((0, target_size[0], target_size[1], 3), dtype=np.float32)
    return np.array(arr, dtype=np.float32)

# --------------------------
# Preprocess for Xception
# --------------------------
def preprocess_frames_for_xception(frames):
    """frames: numpy array (N,H,W,3) dtype float32"""
    return xception_preprocess(frames)  # scales to [-1,1] expected by Xception

# --------------------------
# Audio: mel spectrogram
# --------------------------
def audio_to_melspectrogram(wav_path, sr=16000, n_mels=128, hop_length=512, n_fft=2048, duration=8.0):
    """
    Load audio and return log-mel spectrogram (n_mels x time) in dB.
    duration: seconds to load (trim long files for speed)
    """
    y, sr = librosa.load(wav_path, sr=sr, duration=duration)
    if y is None or len(y) == 0:
        return None
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def spec_to_rgb_image(spec, out_size=(224,224)):
    """
    Convert single-channel spectrogram (2D) to 3-channel uint8 image
    suitable as input to a CNN expecting RGB images.
    """
    s = spec - np.min(spec)
    s = s / (np.max(s) + 1e-8)
    s = (s * 255).astype(np.uint8)
    img = Image.fromarray(s)
    img = img.convert("RGB").resize(out_size)
    return np.array(img)

# --------------------------
# Grad-CAM (TF Keras)
# --------------------------
def find_last_conv_layer(model):
    """
    Try to find the last conv layer name in a Keras model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name or 'sepconv' in layer.name:
            return layer.name
    raise ValueError("No convolutional layer found in the model")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: preprocessed image array (1,H,W,3)
    model: tf.keras.Model
    last_conv_layer_name: string
    pred_index: class index to compute Grad-CAM for (None uses argmax)
    returns: heatmap HxW normalized 0-1
    """
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    # compute channel-wise mean of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    return heatmap

def overlay_heatmap_on_image(img_rgb, heatmap, alpha=0.4, cmap='jet'):
    """
    img_rgb: HxWx3 uint8
    heatmap: small 2D float array 0..1
    returns overlayed uint8 image
    """
    import cv2
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    colormap = cm.get_cmap(cmap)
    heatmap_color = colormap(heatmap_resized)
    heatmap_color = (heatmap_color[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_rgb.astype(np.uint8), 1-alpha, heatmap_color.astype(np.uint8), alpha, 0)
    return overlay

# --------------------------
# Small helper: save numpy array to PNG bytes
# --------------------------
def array_to_png_bytes(arr):
    img = Image.fromarray(arr.astype('uint8'))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()
