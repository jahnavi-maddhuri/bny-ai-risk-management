import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler
import spacy
import math
from openai import OpenAI  # or your preferred LLM client

# -----------------------------
# Setup NLP & LLM
# -----------------------------
nlp = spacy.load("en_core_web_sm")
client = OpenAI(api_key="YOUR_API_KEY")  # replace with your key

# -----------------------------
# Event types
# -----------------------------
EVENT_TYPES = [
    "liquidity_stress", "credit_downgrade", "default", "regulatory_action",
    "management_resignation", "fraud", "market_loss", "merger_acquisition"
]

# -----------------------------
# Step 1: Extract entities
# -----------------------------
def extract_entities(article_summary):
    doc = nlp(article_summary)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "MONEY"]:
            entities.append({
                "entity": ent.text,
                "entity_type": ent.label_,
                "confidence": ent.kb_id_ if hasattr(ent, "kb_id_") else 0.9
            })
    return entities

# -----------------------------
# Step 2: Use LLM to classify event type
# -----------------------------
def classify_event_llm(summary, entity):
    prompt = f"""
    Here are possible bank event types: {EVENT_TYPES}.
    For the following news about a bank, pick the most relevant event type(s):
    News: "{summary}"
    Entity: "{entity}"
    Return the event type(s) as a JSON array.
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    try:
        content = response.choices[0].message.content
        event_list = eval(content)  # expect JSON array like ["liquidity_stress"]
        return event_list
    except:
        return []

# -----------------------------
# Step 3: Compute anomaly factor
# -----------------------------
def compute_anomaly_factor(summaries):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(summaries)
    distances = cosine_distances(X)
    avg_distance = distances.mean(axis=1)
    scaler = MinMaxScaler(feature_range=(1, 2))
    anomaly_factor = scaler.fit_transform(avg_distance.reshape(-1, 1)).flatten()
    return anomaly_factor

# -----------------------------
# Step 4: Compute risk scores
# -----------------------------
def compute_risk_scores(articles_df):
    all_rows = []
    summaries = articles_df['summary'].tolist()
    anomaly_factors = compute_anomaly_factor(summaries)
    
    for idx, summary in enumerate(summaries):
        entities = extract_entities(summary)
        
        for e in entities:
            event_types = classify_event_llm(summary, e['entity'])
            if not event_types:
                continue
            
            for ev in event_types:
                # Severity map
                severity_map = {
                    "liquidity_stress": 8,
                    "credit_downgrade": 7,
                    "default": 10,
                    "regulatory_action": 6,
                    "management_resignation": 5,
                    "fraud": 9,
                    "market_loss": 6,
                    "merger_acquisition": 4
                }
                severity = severity_map.get(ev, 5)
                
                all_rows.append({
                    "entity": e['entity'],
                    "event_type": ev,
                    "confidence": e['confidence'],
                    "anomaly_factor": anomaly_factors[idx],
                    "source": articles_df.loc[idx, 'source'],
                    "summary": summary,
                    "severity": severity
                })
    
    risk_df = pd.DataFrame(all_rows)
    
    # -----------------------------
    # Aggregate per entity-event
    # -----------------------------
    agg_df = risk_df.groupby(['entity', 'event_type', 'severity']).agg({
        'confidence': 'mean',
        'anomaly_factor': 'mean',
        'source': lambda x: list(set(x)),
        'summary': 'first'
    }).reset_index()
    
    # Calculate num_sources
    agg_df['num_sources'] = agg_df['source'].apply(len)
    
    # Compute final risk score
    agg_df['risk_score'] = agg_df.apply(
        lambda row: row['confidence'] * row['severity'] * (1 + math.log(1 + row['num_sources'])) * row['anomaly_factor'],
        axis=1
    )
    
    return agg_df

# -----------------------------
# Step 5: Example usage
# -----------------------------
if __name__ == "__main__":
    data = [
        {"summary": "Bank ABC faces liquidity stress after sudden withdrawals.", "source": "Reuters"},
        {"summary": "Bank XYZ downgraded by credit agencies.", "source": "Bloomberg"},
        {"summary": "Bank ABC management resignation raises concerns.", "source": "Financial Times"},
        {"summary": "Bank XYZ merger acquisition completes successfully.", "source": "Reuters"},
        {"summary": "Bank ABC liquidity stress continues amid panic.", "source": "Bloomberg"},
    ]
    
    df_articles = pd.DataFrame(data)
    risk_scores_df = compute_risk_scores(df_articles)
    print(risk_scores_df)
