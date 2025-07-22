# utils/pdf_exporter.py

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
import os

# Register a Unicode-capable font
font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
if os.path.exists(font_path):
    pdfmetrics.registerFont(TTFont("ArialUnicode", font_path))
else:
    # fallback to Helvetica if Arial Unicode is not found
    font_path = None

def generate_pdf(text: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    if font_path:
        c.setFont("ArialUnicode", 10)
    else:
        c.setFont("Helvetica", 10)

    width, height = letter
    y = height - 40
    line_height = 14

    for line in text.split('\n'):
        if y < 40:
            c.showPage()
            y = height - 40
            c.setFont("ArialUnicode" if font_path else "Helvetica", 10)
        c.drawString(40, y, line)
        y -= line_height

    c.save()
    buffer.seek(0)
    return buffer.read()
