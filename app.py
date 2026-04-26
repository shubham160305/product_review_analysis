import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pdfplumber
import pandas as pd

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Fashion Review Analyzer", layout="centered")

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    ).to(device)

    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer, device

model, tokenizer, device = load_model()

# =============================
# PDF EXTRACTION
# =============================
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# =============================
# SENTIMENT PREDICTION
# =============================
def predict_sentiment(text):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    return pred, probs.cpu().numpy()[0]

# =============================
# FASHION ASPECTS
# =============================
aspects = {
    "Quality": ["quality", "material", "fabric", "durable"],
    "Fit": ["fit", "size", "tight", "loose"],
    "Comfort": ["comfortable", "soft", "itchy"],
    "Style": ["style", "design", "look", "fashion"],
    "Price": ["price", "cheap", "expensive", "worth"],
    "Delivery": ["delivery", "late", "fast", "shipping"]
}

def analyze_aspects(text, sentiment):
    text = text.lower()
    scores = {}

    for aspect, keywords in aspects.items():
        score = sum(word in text for word in keywords)

        # sentiment adjustment
        if sentiment == 0:
            score = -score

        scores[aspect] = score

    return scores

# =============================
# RATING
# =============================
def get_rating(sentiment):
    return [2, 3, 5][sentiment]

# =============================
# UI
# =============================
st.title("Fashion Review Analyzer")

option = st.radio("Choose Input Type:", ["Text Input", "Upload PDF"])

text = ""

# TEXT INPUT
if option == "Text Input":
    text = st.text_area("✍️ Enter your fashion review:")

# PDF INPUT
else:
    uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        st.subheader("📄 Extracted Text Preview")
        st.write(text[:500] + "...")

# =============================
# ANALYSIS
# =============================
if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter or upload text")
    else:
        sentiment, probs = predict_sentiment(text)

        labels = ["Negative 😠", "Neutral 😐", "Positive 😊"]

        # SENTIMENT
        st.subheader("📊 Sentiment")
        st.success(labels[sentiment])

        # ASPECT ANALYSIS
        aspect_scores = analyze_aspects(text, sentiment)

        st.subheader("📌 Aspect Analysis")

        df_aspect = pd.DataFrame({
            "Aspect": list(aspect_scores.keys()),
            "Score": list(aspect_scores.values())
        })

        st.bar_chart(df_aspect.set_index("Aspect"))

        # RATING
        rating = get_rating(sentiment)

        st.subheader("⭐ Overall Rating")
        st.write("⭐" * rating)

        # PROBABILITY
        st.subheader("📈 Sentiment Distribution")

        st.bar_chart({
            "Negative": probs[0],
            "Neutral": probs[1],
            "Positive": probs[2]
        })