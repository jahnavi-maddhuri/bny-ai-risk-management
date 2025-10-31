import pandas as pd
import ollama
import json
import re
from typing import Dict, Any, Union

# --- Configuration: Dynamic Portfolio and Helper Functions ---

# The 10 industries in the portfolio. 
PORTFOLIO_INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "Energy",
    "Manufacturing", "Transportation", "Retail",
    "Telecommunications", "Real Estate", "Agriculture"
]

# Hypothetical Portfolio Holdings (Sum should equal 100%)
HYPOTHETICAL_PORTFOLIO = {
    "Technology": {"stock": "AAPL", "weight": 15},
    "Finance": {"stock": "JPM", "weight": 12},
    "Healthcare": {"stock": "PFE", "weight": 10},
    "Energy": {"stock": "XOM", "weight": 8},
    "Manufacturing": {"stock": "MMM", "weight": 10},
    "Transportation": {"stock": "CSX", "weight": 8},
    "Retail": {"stock": "WMT", "weight": 10},
    "Telecommunications": {"stock": "VZ", "weight": 7},
    "Real Estate": {"stock": "SPG", "weight": 10},
    "Agriculture": {"stock": "DE", "weight": 10},
}

# ⭐️ Using the requested model
OLLAMA_MODEL = "gemma3:1b" 

# File path for reading the input data
FILE_PATH = "data/jpm_risk_assessment_output.csv"

def clean_industry_name(industry_name: str, suffix: str) -> str:
    """Converts 'Real Estate' to 'Real_Estate_risk_score' format expected in the CSV."""
    return industry_name.replace(" ", "_").replace("&", "and").replace("/", "") + suffix

# --- Core Function to Generate Recommendation ---

# --- Helper to Aggregate and Call the LLM Once per Interval ---
def get_interval_recommendation(time_label: str, summaries: list[str], portfolio: dict) -> pd.Series:
    """
    Given all event summaries in one time window (e.g., '1 hour'),
    ask the LLM for a single consolidated recommendation.
    """
    combined_news = "\n\n".join([f"- {s}" for s in summaries if isinstance(s, str)])
    
    prompt = f"""
You are a senior risk strategist advising a client with the following portfolio. 
Your job is to provide one consolidated recommendation for the {time_label} window,
based on all the news events listed below.

Each recommendation should include:
- 'recommendation_type': 'NO ACTION NEEDED', 'REBALANCE', or 'HEDGE/INCREASE'
- 'portfolio_action': specific trade changes (or 'None')
- 'reasoning': concise explanation

Additionally, include:
- 'expected_return_without_action' (percentage)
- 'expected_return_with_action' (percentage)
- 'potential_gain' = (with_action - without_action)

**News events in this {time_label} period:**
{combined_news}

**Portfolio Context:**
{json.dumps(portfolio, indent=2)}

Return only a valid JSON object like this:
{{
"time_window": "{time_label}",
  "recommendation_type": "...",
  "portfolio_action": "...",
  "reasoning": "...",
  "expected_return_without_action": "...",
  "expected_return_with_action": "...",
  "potential_gain": "..."
}}
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"]

        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            raise ValueError("No valid JSON found in model output.")
        result = json.loads(json_match.group(0))

        return pd.Series({
            "time_window": time_label,
            "recommendation_type": result.get("recommendation_type", "UNKNOWN"),
            "portfolio_action": result.get("portfolio_action", "None"),
            "reasoning": result.get("reasoning", "No reasoning."),
            "expected_return_without_action": result.get("expected_return_without_action", "N/A"),
            "expected_return_with_action": result.get("expected_return_with_action", "N/A"),
            "potential_gain": result.get("potential_gain", "N/A")
        })

    except Exception as e:
        print(f"❌ Error generating recommendation for {time_label}: {e}")
        return pd.Series({
            "time_window": time_label,
            "recommendation_type": "ERROR",
            "portfolio_action": "Check logs",
            "reasoning": str(e),
            "expected_return_without_action": "N/A",
            "expected_return_with_action": "N/A",
            "potential_gain": "N/A"
        })


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    df = pd.read_csv(FILE_PATH)
    if "summary" not in df.columns:
        raise ValueError("Missing 'summary' column.")
    if "published_utc" not in df.columns:
        raise ValueError("Missing 'published_utc' column for grouping.")

    df["published_utc"] = pd.to_datetime(df["published_utc"])

    # Define your target time intervals (hours)
    intervals = [1, 6, 24, 48]
    interval_results = []

    for hours in intervals:
        # Select news within this window (example: relative to latest time in data)
        max_time = df["published_utc"].max()
        window_start = max_time - pd.Timedelta(hours=hours)
        mask = (df["published_utc"] >= window_start) & (df["published_utc"] <= max_time)
        summaries_in_window = df.loc[mask, "summary"].tolist()

        print(f"\n🕒 Aggregating {len(summaries_in_window)} news items for {hours}-hour window...")
        result = get_interval_recommendation(f"{hours}-hour", summaries_in_window, HYPOTHETICAL_PORTFOLIO)
        interval_results.append(result)

    # Combine all interval recommendations into a single DataFrame
    final_df = pd.DataFrame(interval_results)
    print("\n✅ Final interval-level recommendations:")
    print(final_df)

    final_df.to_csv("portfolio_interval_recommendations.csv", index=False)
    print("\n💾 Saved to 'portfolio_interval_recommendations.csv'")

