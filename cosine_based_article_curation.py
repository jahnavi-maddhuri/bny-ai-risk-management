import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# =========================
# Load FinBERT model
# =========================
print("Loading FinBERT model...")
model = SentenceTransformer("ProsusAI/finbert")

# =========================
# Load BNY baseline document
# =========================
with open("bnybase.txt", "r", encoding="utf-8") as f:
    bny_baseline_text = f.read()

# =========================
# Compute baseline embeddings
# =========================
print("Encoding baseline...")
bny_embedding = model.encode(bny_baseline_text, normalize_embeddings=True)

# Lexical baseline using TF-IDF
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=8000)
tfidf.fit([bny_baseline_text])
bny_tfidf = tfidf.transform([bny_baseline_text])

# =========================
# Similarity functions
# =========================
def semantic_similarity(article_text):
    article_embedding = model.encode(article_text, normalize_embeddings=True)
    return float(cosine_similarity(article_embedding.reshape(1, -1),
                                   bny_embedding.reshape(1, -1))[0, 0])

def bny_relevance_score(article_text, sem_weight=0.7, lex_weight=0.3):
    """Composite relevance score combining semantic and lexical similarity."""
    sem = semantic_similarity(article_text)
    return {
        "semantic_similarity": round(sem, 4),
    }

# =========================
# Example usage
# =========================
article_example = """
A BNY Cloud-based data share is a tool in a data consumption solution that aggregates data from one cloud-based account with other data consumers via subscription or authentication.
"""

result = bny_relevance_score(article_example)
print("Example Article Relevance:", result)

# =========================
# Batch scoring from CSV
# =========================
def score_articles_from_csv(csv_path, text_col="title", output_csv=None):
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"CSV must contain a '{text_col}' column")

    results = []
    for idx, row in df.iterrows():
        article_text = str(row[text_col])
        scores = bny_relevance_score(article_text)
        results.append({
            "row_index": idx,
            text_col: article_text,
            **scores
        })

    df_results = pd.DataFrame(results)
    if output_csv:
        df_results.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    return df_results
