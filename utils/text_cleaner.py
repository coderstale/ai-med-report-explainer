import re

def clean_ocr_output(text: str) -> str:
    # Replace multiple newlines with space
    text = re.sub(r'\n+', ' ', text)
    # Replace weird unicode characters and symbols
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove unnecessary punctuation clutter
    text = re.sub(r'[^\w\s\-\.\:%/]', '', text)
    # Fix common OCR misreads
    text = text.replace("T.L.C", "Total Leukocyte Count")
    text = text.replace("x101", "x10")
    text = text.replace("x107", "x10")
    # Collapse spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()
