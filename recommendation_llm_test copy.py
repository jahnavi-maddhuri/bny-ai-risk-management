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

def get_portfolio_recommendation(row: pd.Series, portfolio: Dict[str, Any]) -> pd.Series:
    """
    Analyzes an entire DataFrame row and generates a structured portfolio 
    recommendation (type, action, reasoning) using an LLM, returned as a Series.
    """
    # Safely get the news summary
    news_summary = row.get('summary', 'No summary available.')
    
    # 1. Construct the detailed portfolio and risk context
    portfolio_context = "--- Current Portfolio Holdings and Risk Assessment ---\n"
    
    """for industry, details in portfolio.items():
        # Get risk score and justification columns dynamically
        risk_col = clean_industry_name(industry, "_risk_score")
        justification_col = clean_industry_name(industry, "_justification")
        
        # Safely extract data
        risk_score = row.get(risk_col, 'N/A')
        justification = row.get(justification_col, 'No justification provided.')

        score_str = str(risk_score) if pd.notna(risk_score) else "N/A"
        
        portfolio_context += (
            f"- **{industry}** ({details['stock']} - {details['weight']}% weight)\n"
            f"  -> Risk Score (1-10): {score_str}\n"
            f"  -> Justification: {justification}\n"
        )
    """
    # 2. Construct the full LLM prompt demanding JSON output
    prompt = f"""
Today is September 16th, 9 PM. You are a senior risk strategist advising a client with the following portfolio.
Your task is to provide a single, consolidated action recommendation based on all the news events, sharing a percentage estimate of portfolio returns without the action, and with the action. Then you would also share the\
potential gains, which is the difference between the percentage change with action and percentage change without action.\
Your job is to do this for each time interval provided: 1 hour, 6 hours, 24 hours, 48 hours. Each recommendation and associated percentages should be its own row, with separate columns for the\
recommendation and each percentage.

**The News Context:**
"{news_summary}"

**The Portfolio Context:**
{portfolio_context}

**Instruction:**
You MUST return ONLY a valid JSON object matching the following structure.
- 'recommendation_type' must be 'NO ACTION NEEDED', 'REBALANCE', or 'HEDGE/INCREASE'.
- 'portfolio_action' must detail the specific change (e.g., 'Reduce XOM by 5% and Increase AAPL by 5%') or 'None'.
- 'reasoning' must provide a concise justification for the decision.

Remember that if your recommendation_type is 'NO ACTION NEEDED', the portfolio_action should be 'None'.

Expected JSON Format:
{{
    "recommendation_type": "...",
    "portfolio_action": "...",
    "reasoning": "..."
}}
"""
    
    # Define default result series for error handling
    error_result = pd.Series({
        'recommendation_type': 'ERROR', 
        'portfolio_action': 'Check Log', 
        'reasoning': f'LLM/Parsing error for row with summary: {news_summary[:50]}...'
    })

    try:
        # 3. Call Ollama API
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"]

        # Extract and clean JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            raise ValueError("No valid JSON object found in model output.")
        
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_match.group(0))
        result = json.loads(json_str)
        
        # 4. Return structured result as a Series
        return pd.Series({
            'recommendation_type': result.get('recommendation_type', 'UNKNOWN'), 
            'portfolio_action': result.get('portfolio_action', 'None Specified'), 
            'reasoning': result.get('reasoning', 'No reasoning provided.'),
        })
    
    except Exception as e:
        print(f"❌ Error processing row (LLM/JSON): {e}")
        return error_result


# --- 5. Main Execution ---

if __name__ == "__main__":
    
    # 🌟 STEP 1: Load the DataFrame from the specified CSV file
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"✅ Successfully loaded data from {FILE_PATH}")
        print(f"Data shape: {df.shape}")
        
        if 'summary' not in df.columns:
            raise ValueError("The required column 'summary' is missing from the CSV file.")

    except FileNotFoundError:
        # Provide a simple error handler for missing file
        print(f"❌ Error: The file {FILE_PATH} was not found.")
        print("Please ensure the file exists at the specified path.")
        exit()
    except Exception as e:
        print(f"❌ Error loading or validating data: {e}")
        exit()
        
    # Print a sample of the input data for verification
    sample_cols = ['summary'] + [clean_industry_name(i, "_risk_score") for i in PORTFOLIO_INDUSTRIES]
    print("\n--- Sample of Input Data (Summary and Risk Scores) ---")
    print(df[sample_cols].head())
    
    # 🌟 STEP 2: Use df.apply(..., axis=1) to process each row
    print("\n--- Applying Portfolio Action Analysis (Calling LLM on each row) ---\n")
    
    # The result of apply will be a DataFrame where each column is one of the keys
    # returned by the get_portfolio_recommendation function (as a Series).
    recommendation_results = df.apply(
        lambda row: get_portfolio_recommendation(row, portfolio=HYPOTHETICAL_PORTFOLIO), 
        axis=1
    )
    
    # Merge the results back into the original DataFrame
    df = pd.concat([df, recommendation_results], axis=1)

    # 🌟 STEP 3: Display and Save the results
    
    output_cols = ['summary', 'recommendation_type', 'portfolio_action', 'reasoning']
    print("\n--- Final DataFrame with Structured Recommendations (First 5 Rows) ---")
    print(df[output_cols].head())

    OUTPUT_FILE = "portfolio_actions_final_recommendation2.csv"
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Results saved to {OUTPUT_FILE}")
