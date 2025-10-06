# sentiment_gen.py
from transformers import pipeline, set_seed
import time

# Default model: distilgpt2 (lighter and usually less "weird" than gpt2)
DEFAULT_MODEL = "distilgpt2"

# Initialize pipelines lazily in a helper so Streamlit import doesn't hang too long
_pipes = {"sentiment": None, "textgen": {}}

def get_sentiment_pipe():
    if _pipes["sentiment"] is None:
        _pipes["sentiment"] = pipeline("sentiment-analysis")
    return _pipes["sentiment"]

def get_textgen_pipe(model_name):
    if model_name not in _pipes["textgen"]:
        # create and cache a pipeline for the requested model
        _pipes["textgen"][model_name] = pipeline("text-generation", model=model_name)
    return _pipes["textgen"][model_name]

def detect_sentiment(prompt_text, neutral_threshold=0.55):
    """
    Returns (label, score) where label in {"positive","negative","neutral"}.
    """
    if not prompt_text or not prompt_text.strip():
        return "neutral", 1.0
    pipe = get_sentiment_pipe()
    res = pipe(prompt_text[:512])[0]
    label = res["label"].lower()
    score = float(res["score"])
    if score < neutral_threshold:
        return "neutral", score
    return label, score

def _clean_generated(text):
    # Keep text up to first double newline or reasonable cutoff
    if not text:
        return ""
    # If generation included the instruction, remove common markers
    for marker in ["Paragraph:", "OUTPUT:", "TEXT:"]:
        if marker in text:
            text = text.split(marker, 1)[1].strip()
    # Stop at first double newline for single paragraph
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0].strip()
    # Trim to reasonable length
    return text.strip()

def generate_sentiment_text(prompt_text,
                            sentiment_label,
                            model_name=DEFAULT_MODEL,
                            max_length=200,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.95,
                            num_return_sequences=1,
                            strict_mode=False):
    """
    Generate text with a clearer instruction for sentiment alignment.
    strict_mode adds explicit guardrails in the instruction.
    """
    sentiment_label = sentiment_label.lower()
    if sentiment_label not in ("positive", "negative", "neutral"):
        sentiment_label = "neutral"

    # Build a strong instruction so model follows sentiment
    base = f"Write a single paragraph that is clearly {sentiment_label} in tone about: {prompt_text}."
    if strict_mode:
        # more explicit constraints to avoid contradictions
        base = (f"Write a single paragraph that is clearly {sentiment_label} in tone about: {prompt_text}. "
                "Make every sentence reflect that tone. Do not include contradictory or neutral statements. "
                "Keep it coherent and human-like.")

    # Prepend a short instruction label to help some models separate instruction from output
    instruction = f"Instruction: {base}\n\nParagraph:"

    pipe = get_textgen_pipe(model_name)
    # call pipe with sampling params
    gen = pipe(
        instruction,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    outputs = []
    for item in gen:
        txt = item.get("generated_text", "")
        cleaned = _clean_generated(txt)
        # If cleaned is empty, fallback to raw text truncated
        if not cleaned:
            cleaned = txt.strip()[:max_length*2]
        outputs.append(cleaned)
    return outputs
