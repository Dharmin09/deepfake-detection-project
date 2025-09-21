# src/app_streamlit.py
import streamlit as st
import tempfile
import os

from src.infer import load_models, infer_video, infer_audio, combine_scores
from src.generate_report import SimpleReport
from src.config import VIDEO_MODEL_PATH, AUDIO_MODEL_PATH, REPORTS_DIR

st.set_page_config(page_title="RevealAI", layout="wide")
st.title("RevealAI — Deepfake Detection (Demo)")

uploaded = st.file_uploader("Upload a video (.mp4) or audio (.wav)", type=['mp4', 'wav'])

if uploaded is not None:
    # Save temp file
    suffix = os.path.splitext(uploaded.name)[1]
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmpf.write(uploaded.read())
    tmpf.flush()
    path = tmpf.name
    st.info(f"Saved temp file: {path}")

    # Load trained models
    vm, am = load_models(video_path=VIDEO_MODEL_PATH, audio_path=AUDIO_MODEL_PATH)

    res_audio = {}
    res = {'video_score': 0.0, 'heatmaps': []}
    audio_score = None

    if uploaded.type == "video/mp4":
        st.write("🎥 Running video inference...")
        res = infer_video(path, every_n_frames=15, max_frames=20, heatmap_frames=3)
        st.write("Video fake score:", round(res['video_score'], 3))
        for i, img in enumerate(res['heatmaps']):
            st.image(img, caption=f"Heatmap {i+1}", width=300)

    elif uploaded.type == "audio/wav":
        st.write("🎙️ Running audio inference...")
        res_audio = infer_audio(path)
        st.write("Audio fake score:", round(res_audio['audio_score'], 3))
        if res_audio['spec_img'] is not None:
            st.image(res_audio['spec_img'], caption="Spectrogram", width=400)
        audio_score = res_audio['audio_score']

    # Combine scores
    final = combine_scores(res['video_score'], audio_score)
    st.write("🔎 Final combined score:", round(final, 3))

    # PDF report
    if st.button("📄 Generate PDF report"):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        out_path = os.path.join(REPORTS_DIR, "revealai_report.pdf")
        rep = SimpleReport(out_path=out_path)
        rep.add_cover()
        rep.add_result(
            uploaded.name,
            res['video_score'],
            audio_score or 0.0,
            final,
            res['heatmaps'],
            res_audio.get('spec_img')
        )
        pdf_path = rep.output()
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download report", f, file_name="revealai_report.pdf")
