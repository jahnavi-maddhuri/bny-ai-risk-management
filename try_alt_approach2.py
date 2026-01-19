import json
import math
import spacy
import ollama
import numpy as np
import re
import yfinance as yf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

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
def compute_anomaly_factor(summaries, do_clustering=True):
    if len(summaries) == 1 or not do_clustering:
        # single summary: simple TF-IDF distance
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(summaries)
        distances = cosine_distances(X)
        avg_distance = distances.mean(axis=1)
        scaler = MinMaxScaler(feature_range=(1, 2))
        return scaler.fit_transform(avg_distance.reshape(-1, 1)).flatten()
    
    # Multiple summaries: cluster by industry first
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(summaries)
    
    # Elbow method for k
    distortions = []
    K_range = range(1, min(10, len(summaries))+1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5).fit(X)
        distortions.append(kmeans.inertia_)
    # choose k with max elbow approximation
    k = 1 if len(K_range)==1 else np.argmax(np.diff(distortions)) + 2
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5).fit(X)
    cluster_labels = kmeans.labels_
    
    # Compute anomaly factor as distance to cluster centroid
    closest, distances = pairwise_distances_argmin_min(X, kmeans.cluster_centers_[cluster_labels])
    scaler = MinMaxScaler(feature_range=(1, 2))
    return scaler.fit_transform(distances.reshape(-1, 1)).flatten()

# -----------------------------
# Source counting (semantic)
# -----------------------------
def compute_num_sources(entity, summaries):
    return sum(entity.lower() in s.lower() for s in summaries)

# -----------------------------
# Market factor
# -----------------------------

def get_market_factor(ticker, return_components=False):
    """
    Compute the market factor for a given ticker using:
    MF = 0.4*volatility + 0.3*leverage + 0.3*liquidity

    If return_components=True, also returns the individual values.
    """
    try:
        data = yf.Ticker(ticker).info
        volatility = float(data.get("beta", 1.0))        # beta as proxy for volatility
        leverage = float(data.get("debtToEquity", 0.5))  # leverage
        liquidity = float(data.get("currentRatio", 1.0)) # liquidity
    except Exception:
        # Defaults if data unavailable
        volatility, leverage, liquidity = 1.0, 0.5, 1.0

    mf = 0.4 * volatility + 0.3 * leverage + 0.3 * liquidity

    if return_components:
        return mf, volatility, leverage, liquidity
    else:
        return mf


# -----------------------------
# Risk score computation (1-10 normalized)
# -----------------------------
def compute_risk_score(confidence, event_type, num_sources, anomaly_factor, market_factor):
    severity = EVENT_SEVERITY.get(event_type, 0.1)
    raw_score = confidence * severity * (1 + math.log(1 + num_sources)) * anomaly_factor * market_factor
    normalized_score = (raw_score / MAX_POSSIBLE_SCORE) * 10
    normalized_score = min(max(normalized_score, 1), 10)
    return round(normalized_score, 2), round(raw_score, 3), float(confidence), float(severity), int(num_sources), float(anomaly_factor), float(market_factor)

# -----------------------------
# Entity to ticker mapping (with fuzzy matching)
# -----------------------------
ENTITY_TICKER = {
    "Bank ABC": "ABC",
    "Bank XYZ": "XYZ",
    "Bank DEF": "DEF",
    "Bank GHI": "GHI",
    "Bank JKL": "JKL",
    "Google": "GOOGL",
    "JPMorgan Chase": "JPM"
}
from difflib import get_close_matches
def map_entity_to_ticker(entity_name):
    exact = ENTITY_TICKER.get(entity_name)
    if exact:
        return exact
    matches = get_close_matches(entity_name, ENTITY_TICKER.keys(), n=1, cutoff=0.6)
    return ENTITY_TICKER[matches[0]] if matches else None

# -----------------------------
# Main pipeline
# -----------------------------
def score_articles(summaries):
    if isinstance(summaries, str):
        summaries = [summaries]
    
    do_clustering = len(summaries) > 1
    anomaly_factors = compute_anomaly_factor(summaries, do_clustering=do_clustering)
    temp_results = []

    for idx, summary in enumerate(summaries):
        print(f"Processing summary {idx+1}/{len(summaries)}")
        entities = extract_entities(summary)
        for ent in entities:
            ticker = map_entity_to_ticker(ent["entity"])
            # Get market factor and its components
            if ticker:
                mf, vol, lev, liq = get_market_factor(ticker, return_components=True)
            else:
                mf, vol, lev, liq = 0.85, 1.0, 0.5, 1.0  # default values

            events_with_justifications = classify_event_llm(summary, ent["entity"])
            for event_obj in events_with_justifications:
                event = event_obj["event_type"]
                justification = event_obj.get("justification", "")
                num_sources = compute_num_sources(ent["entity"], summaries)
                
                score, raw, conf, sev, srcs, af, mf_final = compute_risk_score(
                    confidence=ent["confidence"],
                    event_type=event,
                    num_sources=num_sources,
                    anomaly_factor=anomaly_factors[idx],
                    market_factor=mf
                )
                
                components = {
                    "confidence": conf,
                    "event_severity": sev,
                    "num_sources": srcs,
                    "anomaly_factor": af,
                    "market_factor": mf_final,
                    "volatility": vol,
                    "leverage": lev,
                    "liquidity": liq,
                    "raw_score": raw
                }

                temp_results.append({
                    "entity": ent["entity"],
                    "entity_type": ent["entity_type"],
                    "ticker": ticker,
                    "event_type": event,
                    "risk_score": score,
                    "components": components,
                    "justification": justification
                })

    # Deduplicate per (entity, event_type)
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
            deduped[key]["components"]["anomaly_factor"] = float(
                np.mean([
                    deduped[key]["components"]["anomaly_factor"],
                    r["components"]["anomaly_factor"]
                ])
            )
            deduped[key]["components"]["market_factor"] = float(
                np.mean([
                    deduped[key]["components"]["market_factor"],
                    r["components"]["market_factor"]
                ])
            )
            # Recompute normalized score
            c = deduped[key]["components"]
            score, raw, conf, sev, srcs, af, mf_final = compute_risk_score(
                confidence=c["confidence"],
                event_type=r["event_type"],
                num_sources=c["num_sources"],
                anomaly_factor=c["anomaly_factor"],
                market_factor=c["market_factor"]
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
        "JPMorgan Chase announced that Marianne Lake, CEO of Consumer & Community Banking, will present at the BancAnalysts Association of Boston Conference on November 4, 2022. She is scheduled to discuss the firm's strategy and outlook. An audio webcast of the presentation and slides will be available on the company's investor relations website for those unable to attend.",
        "JPMorgan Chase & Co. announced it will host its annual Investor Day on **Tuesday, May 21, 2024**, in New York City. Chairman and CEO Jamie Dimon, along with other senior management, will discuss the firm’s businesses, financial performance, and strategic priorities, with a live webcast available on their Investor Relations website.",
        "JPMorgan Chase will host its Third Quarter 2025 earnings call on Friday, October 11, 2024, at 8:30 a.m. Eastern Time. The bank plans to release its financial results an hour earlier at 8:00 a.m. ET, with Chairman and CEO Jamie Dimon and CFO Jeremy Barnum leading the webcast to discuss the results and business outlook. Interested parties can access the live audio webcast and related materials via the company's investor relations website.",
        "JPMorgan Chase & Co. announced that its Board of Directors declared a quarterly dividend on the corporation's common stock. The dividend is set at $1.00 per share. It is payable on October 31, 2023, to stockholders of record as of the close of business on October 6, 2023.",
        "JPMorgan Chase & Co. announced the declaration of regular quarterly cash dividends for numerous series of its outstanding preferred stock. These dividends are generally payable on January 1, 2024, to shareholders of record on various dates in December 2023 and January 2024. This press release outlines the specific dividend amounts and payment details for each preferred stock series.",
        "JPMorgan Chase & Co. announced the declaration of quarterly cash dividends for multiple series of its outstanding preferred stock. These dividends, covering various series including DD, EE, FF, and more, are scheduled for payment on December 15, 2023, to stockholders of record as of December 1, 2023, with specific per-share amounts declared for each series. This is a routine financial announcement regarding distributions to its preferred shareholders."
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
    test_single_event("Google pauses AI image generation feature after controversy")
    
    print("\n=== Multiple news summaries test ===")
    test_multiple_summaries()
