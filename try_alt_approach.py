# TODO:

# - figure out event-entity deduplication
# - dynamic confidence calculated with a bigger transformer model
# - make it faster
# -----------------------------
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

MAX_POSSIBLE_SCORE = 8.0  # realistic max

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
                "confidence": float(ent.kb_id_) if hasattr(ent, "kb_id_") and ent.kb_id_ else 0.9
            })
    return entities

# -----------------------------
# Safe JSON parsing
# -----------------------------
def safe_json_parse(text):
    # Strip <think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if not matches:
        return []
    try:
        return [json.loads(m) for m in matches]
    except Exception:
        return []

# -----------------------------
# Event classification + justification (Ollama)
# -----------------------------
def classify_event_llm(summary, entity):
    prompt = f"""
You are a financial risk analyst.

For the following news summary, identify relevant financial risk events for the entity.  
Return a JSON array of objects. Each object must have:
- "event_type": one of {EVENT_TYPES}
- "justification": a 1-2 sentence explanation for why this event is relevant

News summary:
"{summary}"

Entity:
"{entity}"

Examples:
[{{"event_type": "liquidity_stress", "justification": "The bank faces sudden withdrawals causing liquidity stress."}}]
[]
"""
    response = ollama.chat(
        model="phi4-mini-reasoning",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    raw = response["message"]["content"]
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
# Risk score computation (1-10 normalized)
# -----------------------------
def compute_risk_score(confidence, event_type, num_sources, anomaly_factor):
    severity = EVENT_SEVERITY.get(event_type, 0.1)
    raw_score = confidence * severity * (1 + math.log(1 + num_sources)) * anomaly_factor
    normalized_score = (raw_score / MAX_POSSIBLE_SCORE) * 10
    normalized_score = min(max(normalized_score, 1), 10)
    return round(normalized_score, 2), round(raw_score, 3), confidence, severity, num_sources, anomaly_factor

# -----------------------------
# Main pipeline
# -----------------------------
# -----------------------------
# Main pipeline (deduplicated per event)
# -----------------------------
def score_articles(summaries):
    if isinstance(summaries, str):
        summaries = [summaries]

    anomaly_factors = compute_anomaly_factor(summaries)
    temp_results = []

    # Step 1: compute per-summary event scores
    for idx, summary in enumerate(summaries):
        entities = extract_entities(summary)
        for ent in entities:
            events_with_justifications = classify_event_llm(summary, ent["entity"])
            for event_obj in events_with_justifications:
                event = event_obj["event_type"]
                justification = event_obj.get("justification", "")
                num_sources = compute_num_sources(ent["entity"], summaries)
                score, raw, conf, sev, srcs, af = compute_risk_score(
                    confidence=ent["confidence"],
                    event_type=event,
                    num_sources=num_sources,
                    anomaly_factor=anomaly_factors[idx]
                )

                components = {
                    "confidence": conf,
                    "event_severity": sev,
                    "num_sources": srcs,
                    "anomaly_factor": af,
                    "raw_score": raw
                }

                temp_results.append({
                    "entity": ent["entity"],
                    "entity_type": ent["entity_type"],
                    "event_type": event,
                    "risk_score": score,
                    "components": components,
                    "justification": justification
                })

    # Step 2: deduplicate per (entity, event_type)
    deduped = {}
    for r in temp_results:
        key = (r["entity"], r["event_type"])
        if key not in deduped:
            deduped[key] = r
        else:
            # Aggregate components
            deduped[key]["components"]["num_sources"] = max(
                deduped[key]["components"]["num_sources"],
                r["components"]["num_sources"]
            )
            deduped[key]["components"]["anomaly_factor"] = np.mean([
                deduped[key]["components"]["anomaly_factor"],
                r["components"]["anomaly_factor"]
            ])
            # Recompute normalized score based on updated components
            c = deduped[key]["components"]
            score, raw, conf, sev, srcs, af = compute_risk_score(
                confidence=c["confidence"],
                event_type=r["event_type"],
                num_sources=c["num_sources"],
                anomaly_factor=c["anomaly_factor"]
            )
            deduped[key]["risk_score"] = score
            deduped[key]["components"]["raw_score"] = raw

    return list(deduped.values())

# -----------------------------
# Test wrapper
# -----------------------------
def test_single_event(news_event):
    results = score_articles(news_event)
    if not results:
        print("No events detected.")
    for r in results:
        print(json.dumps(r, indent=2))
    return results

def test_multiple_summaries():
    summaries = [
        "Bank ABC experiences sudden withdrawals leading to liquidity pressure.",
        "Liquidity problems emerge at Bank ABC due to unexpected customer outflows.",
        "Bank ABC's cash reserves are stressed after rapid withdrawal events.",
        "Regulators investigate Bank XYZ over capital adequacy issues.",
        "Bank DEF reports a merger acquisition with Bank GHI.",
        "Bank JKL suffers operational outage affecting trading systems."
    ]
    results = score_articles(summaries)
    for r in results:
        print(json.dumps(r, indent=2))
    return results

# -----------------------------
# Run test cases
# -----------------------------
if __name__ == "__main__":
    print("=== Single event test ===")
    test_single_event("Bank ABC faces liquidity stress after sudden withdrawals.")
    
    print("\n=== Multiple news summaries test ===")
    test_multiple_summaries()
