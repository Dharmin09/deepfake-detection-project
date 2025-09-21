# generate_report.py
# Simple PDF report generator using fpdf

from fpdf import FPDF
from PIL import Image
import os
import tempfile

class SimpleReport:
    def __init__(self, out_path="report.pdf"):
        self.pdf = FPDF()
        self.out_path = out_path
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def add_cover(self, title="RevealAI - Detection Report"):
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=18)
        self.pdf.cell(0, 10, title, ln=True, align='C')
        self.pdf.ln(6)

    def add_result(self, filename, video_score, audio_score, final_score, heatmaps=None, spec_img=None):
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(0, 8, f"File: {filename}", ln=True)
        self.pdf.cell(0, 8, f"Video fake score: {video_score:.3f}", ln=True)
        self.pdf.cell(0, 8, f"Audio fake score: {audio_score:.3f}", ln=True)
        self.pdf.cell(0, 8, f"Final combined score: {final_score:.3f}", ln=True)
        self.pdf.ln(6)

        # Add heatmaps (up to 4 per page nicely)
        if heatmaps:
            self.pdf.set_font("Arial", size=11)
            self.pdf.cell(0, 6, "Video heatmaps:", ln=True)
            for i, img_arr in enumerate(heatmaps):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                Image.fromarray(img_arr.astype('uint8')).save(tmp.name)
                # place two images per row
                w = 90
                x = (i % 2) * (w + 10) + 10
                if i % 2 == 0:
                    self.pdf.ln(2)
                self.pdf.image(tmp.name, x=x, w=w)
                tmp.close()

        # Add spectrogram
        if spec_img is not None:
            self.pdf.add_page()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            Image.fromarray(spec_img.astype('uint8')).save(tmp.name)
            self.pdf.image(tmp.name, w=180)
            tmp.close()

    def output(self):
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        self.pdf.output(self.out_path)
        return self.out_path
