# app.py
import streamlit as st
from sentiment_gen import detect_sentiment, generate_sentiment_text

st.set_page_config(page_title="Sentiment-Aligned Text Generator", layout="centered")
st.title("🧠 Sentiment-Aligned AI Text Generator")
st.write("Enter a prompt and the model will detect sentiment and generate a paragraph matching that sentiment.")

with st.form("prompt_form"):
    prompt = st.text_area("Prompt", height=120, placeholder="Type your prompt here...")
    manual_override = st.selectbox("Manual sentiment (optional)", ["Auto-detect", "Positive", "Negative", "Neutral"])
    length = st.slider("Max tokens/words (approx)", min_value=50, max_value=300, value=120, step=10)
    num_out = st.slider("Number of outputs", min_value=1, max_value=3, value=1)
    submitted = st.form_submit_button("Generate")

if submitted:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Detecting sentiment..."):
            detected, score = detect_sentiment(prompt)
        st.markdown(f"**Detected sentiment:** `{detected}` (confidence: {score:.2f})")
        chosen_sentiment = detected
        if manual_override != "Auto-detect":
            chosen_sentiment = manual_override.lower()
            st.info(f"Manual override: using `{chosen_sentiment}`")

        with st.spinner("Generating text..."):
            outputs = generate_sentiment_text(prompt, chosen_sentiment, max_length=length, num_return_sequences=num_out)

        st.success("Generated!")
        for i, out in enumerate(outputs, 1):
            st.subheader(f"Output #{i}")
            st.write(out)
            st.write("---")

st.sidebar.header("About")
st.sidebar.write("""
- Uses a sentiment classifier to infer prompt sentiment and GPT-2 to generate text.
- You can manually override sentiment and control length/outputs.
""")
st.sidebar.markdown("**Tip:** Keep prompts short and focused for best results.")
