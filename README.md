🧠 Sentiment-Aligned Text Generator

An AI-powered Streamlit web app that generates text aligned with the sentiment (Positive, Negative, or Neutral) of the user’s input prompt.

This project demonstrates Natural Language Processing (NLP) and Text Generation using Hugging Face Transformers.

🚀 Features

Generate AI-based text aligned with sentiment

Choose tone: Positive, Negative, or Neutral

Simple and interactive UI built with Streamlit

Uses GPT-2 model for text generation

100% local execution (no API key needed)

🛠️ Tech Stack

Language: Python

Framework: Streamlit

Libraries: transformers, torch, accelerate, pandas, numpy

⚙️ Installation and Setup
1️⃣ Clone or Create a Folder
mkdir sentiment-generator
cd sentiment-generator

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate    # for Windows
# source venv/bin/activate  # for Mac/Linux

3️⃣ Install Dependencies
pip install streamlit transformers torch accelerate pandas

4️⃣ Create app.py and Paste the Code
import streamlit as st
from transformers import pipeline

st.title("🧠 Sentiment-Aligned Text Generator")
st.write("Generate Positive, Negative, or Neutral text using AI ✨")

# Load model
generator = pipeline("text-generation", model="gpt2")

# User input
prompt = st.text_input("Enter your prompt:")
sentiment = st.selectbox("Choose sentiment:", ["Positive", "Negative", "Neutral"])

# Sentiment templates
sentiment_templates = {
    "Positive": "Write something optimistic and encouraging about: ",
    "Negative": "Write something sad or pessimistic about: ",
    "Neutral": "Write something factual and neutral about: ",
}

if st.button("Generate"):
    input_text = sentiment_templates[sentiment] + prompt
    result = generator(input_text, max_length=100, num_return_sequences=1)
    st.subheader(f"{sentiment} Response:")
    st.write(result[0]['generated_text'])

5️⃣ Run the App
streamlit run app.py

🧩 Example

Prompt:

I have an important interview tomorrow and I'm nervous.


Outputs:

Positive: You’ve got this! Just breathe and trust your preparation.

Negative: I can’t stop worrying; what if I mess everything up?

Neutral: You have an interview tomorrow. Be ready and stay calm.

📚 Learning Outcomes

Understanding of sentiment-based text generation

Hands-on with Streamlit app development

Practical use of Hugging Face Transformers

Prompt engineering and language model fine-tuning basics

💡 Future Enhancements

Add multiple sentiment categories (Joy, Fear, Anger)

Adjust paragraph length and tone strength

Save history of generated responses

Deploy to Streamlit Cloud or Hugging Face Spaces

🤝 Acknowledgements

Streamlit

Hugging Face Transformers

PyTorch

