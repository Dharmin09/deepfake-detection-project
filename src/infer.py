# src/infer.py
import os
import numpy as np
import tempfile
import shutil # <--- FIX: Added for directory cleanup
from tensorflow.keras.models import load_model
from src.config import VIDEO_MODEL_PATH, AUDIO_MODEL_PATH, IMG_SIZE_VIDEO, IMG_SIZE_AUDIO
from src.utils import extract_frames, load_images_to_array, preprocess_frames_for_xception, make_gradcam_heatmap, find_last_conv_layer, overlay_heatmap_on_image, audio_to_melspectrogram, spec_to_rgb_image

# ... load_models function is unchanged ...
_VIDEO_MODEL = None
_AUDIO_MODEL = None

def load_models(video_model_path=VIDEO_MODEL_PATH, audio_model_path=AUDIO_MODEL_PATH):
    global _VIDEO_MODEL, _AUDIO_MODEL
    if _VIDEO_MODEL is None and os.path.exists(video_model_path):
        _VIDEO_MODEL = load_model(video_model_path)
    if _AUDIO_MODEL is None and os.path.exists(audio_model_path):
        _AUDIO_MODEL = load_model(audio_model_path)
    return _VIDEO_MODEL, _AUDIO_MODEL

def infer_video(video_path, every_n_frames=15, max_frames=20, heatmap_frames=3):
    model, _ = load_models()
    if model is None:
        raise FileNotFoundError("Video model not found at path: " + VIDEO_MODEL_PATH)
    
    # FIX: Use try...finally to ensure cleanup
    tmpdir = tempfile.mkdtemp(prefix="revealai_frames_")
    try:
        saved = extract_frames(video_path, tmpdir, every_n_frames=every_n_frames, max_frames=max_frames, resize=IMG_SIZE_VIDEO)
        if saved == 0:
            return {"video_score": 0.0, "heatmaps": []}

        files = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith((".jpg",".png"))])
        frames = load_images_to_array(files, target_size=IMG_SIZE_VIDEO)
        X = preprocess_frames_for_xception(frames)

        preds = model.predict(X, verbose=0)
        fake_idx = 1 if preds.shape[1] > 1 else 0
        fake_probs = preds[:, fake_idx]
        avg_fake_prob = float(np.mean(fake_probs))

        heatmaps = []
        try:
            last_conv = find_last_conv_layer(model)
            for i in range(min(heatmap_frames, len(frames))):
                img = X[i:i+1]
                hm = make_gradcam_heatmap(img, model, last_conv, pred_index=fake_idx)
                overlay = overlay_heatmap_on_image(frames[i].astype('uint8'), hm)
                heatmaps.append(overlay)
        except Exception as e:
            print(f"⚠️ Could not generate heatmaps: {e}")
        
        return {"video_score": avg_fake_prob, "heatmaps": heatmaps}
    finally:
        # This will run whether the function succeeds or fails
        shutil.rmtree(tmpdir)

# ... infer_audio and combine_scores are unchanged ...
def infer_audio(wav_path):
    """
    Returns {'audio_score': float, 'spec_img': np.uint8 image}
    """
    _, model = load_models()
    if model is None:
        raise FileNotFoundError("Audio model not found at path: " + AUDIO_MODEL_PATH)
    spec = audio_to_melspectrogram(wav_path, duration=8.0)
    if spec is None:
        return {"audio_score": 0.0, "spec_img": None}
    img = spec_to_rgb_image(spec, out_size=IMG_SIZE_AUDIO)
    x = img.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0]
    fake_idx = 1 if len(pred) > 1 else 0
    fake_prob = float(pred[fake_idx])
    return {"audio_score": fake_prob, "spec_img": img}

def combine_scores(video_score, audio_score, video_weight=0.6, audio_weight=0.4):
    """
    Weighted average. If audio_score is None, return video_score.
    """
    if audio_score is None:
        return video_score
    return float(video_score * video_weight + audio_score * audio_weight)