# Do not use the ticker for entity mapping, this NEEDS TO BE CHANGED
# Fix the entity detection algorithm
import pandas as pd
import numpy as np
import json
import spacy
import ollama   
import re
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min

# =========================
# ENTITY TO TICKER MAPPING
# =========================
nlp = spacy.load("en_core_web_sm")

ENTITY_TICKER_MAP = {
    "JPMorgan Chase": "JPM",
    "Goldman Sachs": "GS",
    "Bank of America": "BAC",
    "Citigroup": "C",
    "Wells Fargo": "WFC"
}

def map_entity_to_ticker(entity):
    match, score, _ = process.extractOne(
        entity,
        ENTITY_TICKER_MAP.keys(),
        scorer=fuzz.token_sort_ratio
    )
    if score >= 80:
        return ENTITY_TICKER_MAP[match]
    return None

# =========================
# ENTITY EXTRACTION
# =========================

def extract_entities(text):
    entities = []
    for name in ENTITY_TICKER_MAP.keys():
        if name.lower() in text.lower():
            entities.append({
                "entity": name,
                "entity_type": "ORG",
                "confidence": 0.9
            })
    return entities

# =========================
# MARKET FACTOR
# =========================
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
        print("Market data:", volatility, leverage, liquidity)
    except Exception:
        # Defaults if data unavailable
        volatility, leverage, liquidity = 1.0, 0.5, 1.0

    mf = 0.4 * volatility + 0.3 * leverage + 0.3 * liquidity

    if return_components:
        return mf, volatility, leverage, liquidity
    else:
        return mf


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


# =========================
# EVENT CLASSIFICATION
# =========================

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

# =========================
# SOURCE COUNTING
# =========================

def compute_num_sources(entity, summaries):
    return sum(entity.lower() in s.lower() for s in summaries)

# =========================
# ANOMALY FACTOR
# =========================
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def compute_anomaly_factor(summaries, do_clustering=True, plot_clusters=False):
    """
    Compute anomaly factors for a list of text summaries.

    Args:
        summaries (list of str): List of summaries to analyze.
        do_clustering (bool): Whether to cluster summaries (for multi-summary anomaly).
        plot_clusters (bool): Whether to plot the clusters in 2D.

    Returns:
        np.ndarray: Anomaly factor per summary (scaled 1-2).
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(summaries)

    if len(summaries) == 1 or not do_clustering:
        # Single summary or clustering disabled
        distances = cosine_distances(X)
        avg_distance = distances.mean(axis=1)
        scaler = MinMaxScaler(feature_range=(1, 2))
        return scaler.fit_transform(avg_distance.reshape(-1, 1)).flatten()

    # Multiple summaries: cluster summaries
    distortions = []
    K_range = range(1, min(10, len(summaries)) + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5).fit(X)
        distortions.append(kmeans.inertia_)

    # Choose k using approximate elbow (max diff of distortions)
    k = 1 if len(K_range) == 1 else np.argmax(np.diff(distortions)) + 2

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5).fit(X)
    cluster_labels = kmeans.labels_

    # Distance to cluster centroid = anomaly factor
    _, distances_to_centroid = pairwise_distances_argmin_min(X, kmeans.cluster_centers_[cluster_labels])
    scaler = MinMaxScaler(feature_range=(1, 2))
    anomaly_factors = scaler.fit_transform(distances_to_centroid.reshape(-1, 1)).flatten()

    # Optional: plot clusters using PCA
    if plot_clusters:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X.toarray())
        plt.figure(figsize=(8, 6))
        for cluster in range(k):
            idxs = cluster_labels == cluster
            plt.scatter(X_pca[idxs, 0], X_pca[idxs, 1], label=f"Cluster {cluster}", alpha=0.7)
        plt.title("Summary Clusters (PCA projection)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.show()

    return anomaly_factors



# =========================
# RISK SCORING (FIXED)
# =========================
import math

MAX_POSSIBLE_SCORE = 8.0  # conservative upper bound

def compute_risk_score(
    confidence,
    event_type,
    num_sources,
    anomaly_factor,
    market_factor
):
    event_severity = EVENT_SEVERITY.get(event_type, 0.1)

    raw_score = (
        confidence
        * event_severity
        * (1 + math.log(1 + num_sources))
        * anomaly_factor
        * market_factor
    )

    final_score = (raw_score / MAX_POSSIBLE_SCORE) * 10
    final_score = min(max(final_score, 1.0), 10.0)

    return {
        "final_score": round(final_score, 2),
        "raw_score": round(raw_score, 4),
        "confidence": float(confidence),
        "event_severity": float(event_severity),
        "num_sources": int(num_sources),
        "anomaly_factor": float(anomaly_factor),
        "market_factor": float(market_factor)
    }
# =========================
# CORE SCORING FUNCTION (FIXED CONSUMPTION)
# =========================

def score_articles(summaries, ticker_override=None, anomaly_factor=None):
    if isinstance(summaries, str):
        summaries = [summaries]

    temp_results = []

    for idx, summary in enumerate(summaries):
        entities = extract_entities(summary)
        if not entities:
            entities = [{"entity": None, "entity_type": None, "confidence": 0.9}]

        for ent in entities:
            ticker = ticker_override or (map_entity_to_ticker(ent["entity"]) if ent["entity"] else None)
            if ticker:
                mf, vol, lev, liq = get_market_factor(ticker, return_components=True)
            else:
                mf, vol, lev, liq = get_market_factor(None, return_components=True)

            events = classify_event_llm(summary, ent["entity"] or "Unknown")
            for e in events:
                e.setdefault("event_type", "unknown_event")
                e.setdefault("justification", "No justification provided")

            if not events:
                events = [{"event_type": "unknown_event", "justification": "No specific event detected"}]

            for e in events:
                num_sources = compute_num_sources(ent["entity"] or "", summaries)

                event_type = e.get("event_type", "unknown_event")

                score_obj = compute_risk_score(
                    confidence=ent["confidence"],
                    event_type=event_type,
                    num_sources=num_sources,
                    anomaly_factor=anomaly_factor[idx] if anomaly_factor is not None else 1.0,
                    market_factor=mf
                )

                temp_results.append({
                    "entity": ent["entity"],
                    "entity_type": ent["entity_type"],
                    "ticker": ticker,
                    "event_type": event_type,
                    "risk_score": score_obj["final_score"],
                    "components": {
                        "confidence": score_obj["confidence"],
                        "event_severity": score_obj["event_severity"],
                        "num_sources": score_obj["num_sources"],
                        "anomaly_factor": score_obj["anomaly_factor"],
                        "market_factor": score_obj["market_factor"],
                        "volatility": vol,
                        "leverage": lev,
                        "liquidity": liq,
                        "raw_score": score_obj["raw_score"]
                    },
                    "justification": e["justification"]
                })
        print(f"Processed summary {idx+1}/{len(summaries)}")
        print(json.dumps(temp_results[-len(events):], indent=2))

    return temp_results



# =========================
# CSV DRIVER
# =========================
import os

def score_csv_with_details(
    csv_path,
    n=None,
    output_csv=None,
    summary_col="title",
    id_col="id",
    ticker_col="ticker",
    batch_size=10
):
    """
    Reads a CSV, computes anomaly factors for all summaries at once,
    scores each row, appends results every `batch_size` rows,
    and resumes from the last written row if output_csv exists.
    """

    df = pd.read_csv(csv_path)

    if summary_col not in df.columns:
        raise ValueError(f"CSV must contain a '{summary_col}' column")
    if ticker_col not in df.columns:
        df[ticker_col] = None
    if id_col not in df.columns:
        df[id_col] = df.index

    # if n is not None:
    #     df = df.head(n)

    processed_ids = set()
    write_header = True

    if output_csv and os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        if id_col in existing.columns:
            processed_ids = set(existing[id_col].astype(str))
            write_header = False

    df[id_col] = df[id_col].astype(str)
    df = df[~df[id_col].isin(processed_ids)]

    if df.empty:
        return pd.DataFrame()

    summaries = df[summary_col].astype(str).tolist()
    anomaly_factors = compute_anomaly_factor(summaries, do_clustering=True)

    buffer = []

    def flush_to_csv(rows):
        if not rows or not output_csv:
            return
        pd.DataFrame(rows).to_csv(
            output_csv,
            mode="a",
            index=False,
            header=write_header
        )
        rows.clear()

    for pos, (_, row) in enumerate(df.iterrows()):
        summary = str(row[summary_col])
        row_id = row[id_col]
        ticker = (
            row[ticker_col]
            if pd.notna(row[ticker_col]) and str(row[ticker_col]).strip()
            else None
        )

        results = score_articles(
            [summary],
            ticker_override=ticker,
            anomaly_factor=[anomaly_factors[pos]]
        )

        for r in results:
            buffer.append({
                "row_index": pos,
                id_col: row_id,
                summary_col: summary,
                "entity": r["entity"],
                "entity_type": r["entity_type"],
                "ticker": r["ticker"],
                "event_type": r["event_type"],
                "risk_score": r["risk_score"],
                "justification": r["justification"],
                "confidence": r["components"]["confidence"],
                "event_severity": r["components"]["event_severity"],
                "num_sources": r["components"]["num_sources"],
                "anomaly_factor": r["components"]["anomaly_factor"],
                "market_factor": r["components"]["market_factor"],
                "volatility": r["components"]["volatility"],
                "leverage": r["components"]["leverage"],
                "liquidity": r["components"]["liquidity"],
                "raw_score": r["components"]["raw_score"]
            })

        if (pos + 1) % batch_size == 0:
            flush_to_csv(buffer)

    flush_to_csv(buffer)

    return pd.DataFrame()


# =========================
# EXAMPLE RUNS
# =========================

if __name__ == "__main__":
    # print("=== Single event test ===")
    # out = score_articles(
    #     "BMW EV sales roar in Korea on strong lineup of new models - KED Global"
    # )
    # print(json.dumps(out, indent=2))

    df_results = score_csv_with_details("adilusethis.csv", n=10, output_csv="results_adilusethis.csv")

