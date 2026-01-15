import json
import math
import spacy
import ollama
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Load NLP
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Event types and severity
# -----------------------------
EVENT_SEVERITY = {
    "counterparty_default": 1.0,
    "liquidity_stress": 0.9,
    "credit_downgrade": 0.8,
    "regulatory_action": 0.7,
    "fraud_investigation": 0.9,
    "sanctions_exposure": 0.85,
    "operational_outage": 0.6,
    "earnings_miss": 0.4,
    "capital_raise": 0.3,
    "merger_acquisition": 0.2
}

EVENT_TYPES = list(EVENT_SEVERITY.keys())

# -----------------------------
# Entity extraction (spaCy)
# -----------------------------
def extract_entities(summary):
    doc = nlp(summary)
    entities = []

    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE"]:
            entities.append({
                "entity": ent.text,
                "entity_type": ent.label_,
                "confidence": 0.8
            })

    return entities

# -----------------------------
# Safe JSON parsing
# -----------------------------
def safe_json_parse(text):
    # Remove <think> blocks completely
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Find the last JSON array in the text
    matches = re.findall(r"\[[^\]]*\]", text, re.DOTALL)
    if not matches:
        return []

    try:
        return json.loads(matches[-1])
    except Exception:
        return []
def safe_json_parse(text):
    # Remove <think> blocks completely
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Find the last JSON array in the text
    matches = re.findall(r"\[[^\]]*\]", text, re.DOTALL)
    if not matches:
        return []

    try:
        return json.loads(matches[-1])
    except Exception:
        return []


# -----------------------------
# Event classification (Ollama)
# -----------------------------
def classify_event_llm(summary, entity):
    prompt = f"""
You are a financial risk analyst.

Return ONLY a JSON array.
Each element must be one of: {EVENT_TYPES}
Do not include explanations or markdown.

News summary:
"{summary}"

Entity:
"{entity}"

Examples:
["liquidity_stress"]
[]
"""
    response = ollama.chat(
        model="phi4-mini-reasoning",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )

    raw = response["message"]["content"]
    print("LLM", raw)
    return safe_json_parse(raw)

# -----------------------------
# Anomaly factor (TF-IDF)
# -----------------------------
def compute_anomaly_factor(summaries):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(summaries)

    distances = cosine_distances(X)
    avg_distance = distances.mean(axis=1)

    scaler = MinMaxScaler(feature_range=(1, 2))
    return scaler.fit_transform(avg_distance.reshape(-1, 1)).flatten()

# -----------------------------
# Source counting
# -----------------------------
def compute_num_sources(entity, summaries):
    return sum(entity.lower() in s.lower() for s in summaries)

# -----------------------------
# Risk score computation
# -----------------------------
def compute_risk_score(confidence, event_type, num_sources, anomaly_factor):
    severity = EVENT_SEVERITY.get(event_type, 0.1)
    return confidence * severity * (1 + math.log(1 + num_sources)) * anomaly_factor

# -----------------------------
# Main pipeline
# -----------------------------
def score_articles(summaries):
    if isinstance(summaries, str):
        summaries = [summaries]

    anomaly_factors = compute_anomaly_factor(summaries)
    print(anomaly_factors)
    results = []

    for idx, summary in enumerate(summaries):
        entities = extract_entities(summary)
        print(entities)

        for ent in entities:
            events = classify_event_llm(summary, ent["entity"])
            print(events)

            for event in events:
                num_sources = compute_num_sources(ent["entity"], summaries)

                risk = compute_risk_score(
                    confidence=ent["confidence"],
                    event_type=event,
                    num_sources=num_sources,
                    anomaly_factor=anomaly_factors[idx]
                )
                print(risk)

                results.append({
                    "entity": ent["entity"],
                    "entity_type": ent["entity_type"],
                    "event_type": event,
                    "risk_score": round(risk, 3)
                })

    return results

# -----------------------------
# Test wrapper (single event)
# -----------------------------
def test_single_event(news_event: str):
    results = score_articles(news_event)
    print(results)

    if not results:
        print("No events detected.")
        return []

    for r in results:
        print(json.dumps(r, indent=2))

    return results

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    news = "Bank ABC faces liquidity stress after sudden withdrawals."
    test_single_event(news)
