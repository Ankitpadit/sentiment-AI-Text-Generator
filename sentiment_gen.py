# sentiment_gen.py
from transformers import pipeline, set_seed

# Initialize pipelines (first run downloads models)
_sentiment_pipe = pipeline("sentiment-analysis")
_textgen_pipe = pipeline("text-generation", model="gpt2")
set_seed(42)

def detect_sentiment(prompt_text, neutral_threshold=0.55):
    """
    Returns tuple (sentiment_label, confidence)
    sentiment_label in {"positive","negative","neutral"}
    """
    if not prompt_text or not prompt_text.strip():
        return "neutral", 1.0

    res = _sentiment_pipe(prompt_text[:512])[0]
    label = res["label"].lower()
    score = float(res["score"])

    if score < neutral_threshold:
        return "neutral", score
    return label, score

def generate_sentiment_text(prompt_text, sentiment_label, max_length=120, num_return_sequences=1):
    sentiment_label = sentiment_label.lower()
    if sentiment_label not in ("positive", "negative", "neutral"):
        sentiment_label = "neutral"

    instruction = f"Write a {sentiment_label} paragraph about: {prompt_text}\n\nParagraph:"

    gen = _textgen_pipe(instruction,
                        max_length=max_length,
                        num_return_sequences=num_return_sequences,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.8)
    outputs = []
    for item in gen:
        txt = item["generated_text"]
        if "Paragraph:" in txt:
            txt = txt.split("Paragraph:", 1)[1].strip()
        txt = txt.split("\n\n")[0].strip()
        outputs.append(txt)
    return outputs
