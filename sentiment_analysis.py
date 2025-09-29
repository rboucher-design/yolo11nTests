import runpod
from transformers import pipeline


def load_model():
    return pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )


def sentiment_analysis_handler(event):
    global model

    if "model" not in globals():
        model = load_model()

    text = event["input"].get("text")

    if not text:
        return {"error": "No text provided for analysis."}

    result = model(text)[0]

    return {"sentiment": result["label"], "score": float(result["score"])}


runpod.serverless.start({"handler": sentiment_analysis_handler})