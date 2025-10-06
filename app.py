# app.py
import streamlit as st
from sentiment_gen import detect_sentiment, generate_sentiment_text, DEFAULT_MODEL
import time

st.set_page_config(page_title="Sentiment-Aligned Text Generator", layout="centered")
st.title("🧠 Sentiment-Aligned Text Generator")
st.write("Enter a prompt and get a paragraph that matches the sentiment.")

with st.form("form"):
    prompt = st.text_area("Prompt", height=120, placeholder="Type your prompt here...")
    auto_detect = st.checkbox("Auto-detect sentiment (use model)", value=True)
    manual = st.selectbox("Manual sentiment (if not auto)", ["Positive", "Negative", "Neutral"])
    model_choice = st.selectbox("Model (if memory permits)", ["distilgpt2", "gpt2", "gpt2-medium"])
    strict_mode = st.checkbox("Strict mode (force consistent tone)", value=True)
    length = st.slider("Max length (approx tokens)", 50, 350, 200, 10)
    temperature = st.slider("Temperature (creativity)", 0.3, 1.0, 0.7, 0.05)
    top_k = st.slider("top_k", 0, 200, 50, 10)
    top_p = st.slider("top_p (nucleus)", 0.5, 1.0, 0.95, 0.01)
    n_outputs = st.slider("Number of outputs", 1, 3, 1)
    submitted = st.form_submit_button("Generate")

if submitted:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Detecting sentiment..."):
            if auto_detect:
                detected, score = detect_sentiment(prompt)
                st.markdown(f"**Detected sentiment:** `{detected}` (confidence: {score:.2f})")
                sentiment = detected
            else:
                sentiment = manual.lower()
                st.info(f"Using manual sentiment: {sentiment}")

        st.markdown("**Generated text (may take a few seconds on first run)...**")
        with st.spinner("Generating..."):
            outputs = generate_sentiment_text(
                prompt_text=prompt,
                sentiment_label=sentiment,
                model_name=model_choice,
                max_length=length,
                temperature=temperature,
                top_k=top_k if top_k>0 else None,
                top_p=top_p,
                num_return_sequences=n_outputs,
                strict_mode=strict_mode
            )
        for i, out in enumerate(outputs, 1):
            st.subheader(f"Output #{i}")
            st.write(out)
            st.write("---")

st.sidebar.header("Tips")
st.sidebar.write("""
- If output is weird, try:
  - Using **distilgpt2** (lighter) or **gpt2-medium** (better coherence).
  - Increase `max length` to 200-250.
  - Lower `temperature` to 0.5-0.7 to reduce randomness.
  - Turn on `Strict mode`.
- If memory error occurs, choose `distilgpt2`.
""")
