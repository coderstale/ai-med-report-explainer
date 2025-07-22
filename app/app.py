import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import time
from io import BytesIO
from PIL import Image
from difflib import get_close_matches

from ocr.extract_text import extract_text_from_pdf, extract_text_from_image
from explain.explain_results import explain_medical_report
from utils.text_cleaner import clean_ocr_output
from utils.chatbot import get_chat_response
from utils.pdf_exporter import generate_pdf
from utils.charts import generate_cbc_chart
from utils.parser import parse_lab_report
from utils.hf_ner_parser import extract_lab_results_from_ner
from ml.diagnose import predict_condition, predict_condition_from_image  # ✅ include image model

# ----------------- Sidebar -----------------
st.sidebar.title("⚙️ Settings")
model_options = ["llama3", "medllama2"]
model = st.sidebar.selectbox("🧠 Choose AI Model", model_options)
ocr_engine = "tesseract"  # TrOCR removed

# ----------------- Main UI -----------------
st.title("🧪 AI Medical Report Explainer")
uploaded_file = st.file_uploader("Upload a Lab Report (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    file_bytes = uploaded_file.read()

    # Save uploaded file temporarily
    file_ext = uploaded_file.name.split('.')[-1]
    temp_path = f"temp_upload.{file_ext}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    # Show uploaded image if not a PDF
    if file_ext.lower() in ["jpg", "jpeg", "png"]:
        st.subheader("🖼️ Uploaded Image")
        st.image(Image.open(temp_path), caption="Uploaded Report Image", use_column_width=True)

    # OCR
    with st.spinner("🔍 Extracting text using Tesseract..."):
        start_time = time.time()
        if uploaded_file.name.endswith(".pdf"):
            raw_text = extract_text_from_pdf(temp_path)
        else:
            raw_text = extract_text_from_image(temp_path)
        duration = time.time() - start_time

    st.success(f"⏱️ OCR Completed in {duration:.2f} seconds")

    # ----------------- Raw + Cleaned Text -----------------
    st.subheader("📄 Raw Extracted Text")
    st.text_area("OCR Output", raw_text, height=200)

    cleaned_text = clean_ocr_output(raw_text)
    st.subheader("🧼 Cleaned Text")
    st.text_area("Cleaned", cleaned_text, height=200)

    # ----------------- AI Explanation -----------------
    with st.spinner("🧠 Generating AI Explanation..."):
        explanation = explain_medical_report(cleaned_text, model)
    st.subheader("🤖 Explanation")
    st.markdown(explanation)

    # ----------------- PDF Export -----------------
    pdf_data = generate_pdf(explanation)
    st.download_button(
        label="📄 Download Explanation as PDF",
        data=pdf_data,
        file_name="report_explanation.pdf",
        mime="application/pdf",
    )

    # ----------------- Chart -----------------
    st.subheader("📊 Visualization")
    fig = generate_cbc_chart(cleaned_text)
    if fig:
        st.pyplot(fig)
    else:
        st.warning("Couldn't generate chart from text.")

    # ----------------- Parsed Lab Values -----------------
    st.subheader("🧬 Parsed Lab Values (NER)")
    parsed_results = extract_lab_results_from_ner(cleaned_text)
    if not parsed_results:
        st.warning("NER failed to find lab values. Falling back to regex parsing.")
        parsed_results = parse_lab_report(cleaned_text)

    feature_map = {}
    if parsed_results:
        for item in parsed_results:
            label = item["name"].strip().lower()
            val = item["value"]
            unit = item.get("unit", "")
            st.markdown(f"- **{label.title()}**: {val} {unit}")
            try:
                feature_map[label] = float(val)
            except:
                pass
    else:
        st.warning("Still couldn't recognize any lab values.")

    # Debug output
    st.subheader("🧪 Feature Map (for ML Diagnosis)")
    st.code(feature_map, language="json")

    # ----------------- ML-Based Diagnosis -----------------
    st.subheader("🧠 ML-Based Diagnosis")
    required_features = {
        "haemoglobin": None,
        "t.l.c": None,
        "platelet count": None,
        "mcv": None
    }

    for required in required_features:
        if required in feature_map:
            required_features[required] = feature_map[required]
        else:
            match = get_close_matches(required, feature_map.keys(), n=1, cutoff=0.6)
            if match:
                required_features[required] = feature_map[match[0]]

    if all(v is not None for v in required_features.values()):
        features = list(required_features.values())
        diagnosis = predict_condition(features)
        st.success(f"Predicted Condition (Lab Values): **{diagnosis}**")
    else:
        missing = [k for k, v in required_features.items() if v is None]
        st.info(f"Not enough data for diagnosis. Missing: {', '.join(missing)}")

    # ✅ CNN Diagnosis (Image Model)
    if file_ext.lower() in ["jpg", "jpeg", "png"]:
        image = Image.open(temp_path).convert("RGB")
        image_diagnosis = predict_condition_from_image(image)
        st.success(f"Predicted Condition (Image Model): **{image_diagnosis}**")

    # ----------------- Chatbot Q&A -----------------
    st.subheader("💬 Ask AI About Your Report")
    user_question = st.text_input("Ask a question (e.g., What does MCV mean?)")
    if user_question:
        with st.spinner("🤖 Thinking..."):
            answer = get_chat_response(user_question, explanation, model)
        st.markdown(f"**Answer:** {answer}")
