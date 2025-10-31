import pandas as pd
import ollama
import json
import re
import sys
from typing import Dict, Any, Union

# --- Configuration and Constants ---

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

# The model to use for all LLM calls
OLLAMA_MODEL = "gemma3:1b" 

# File path for reading the input data (Now using the aggregated file structure)
FILE_PATH = "portfolio_risk_aggregated_6hr.csv" 

# --- Helper Functions ---

def clean_industry_name(industry_name: str, suffix: str) -> str:
    """Converts 'Real Estate' to 'Real_Estate_period_risk_score' format."""
    # Note: Using '_period_' suffix to match the new input file structure
    return industry_name.replace(" ", "_").replace("&", "and").replace("/", "") + suffix

def clean_llm_json(content: str) -> Dict[str, Any]:
    """Extracts, cleans, and parses the JSON object from the LLM output."""
    json_match = re.search(r'\{[\s\S]*\}', content)
    if not json_match:
        raise ValueError("No valid JSON object found in model output.")
    
    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_match.group(0))
    return json.loads(json_str)

# --- Stage 1: Portfolio Actions (LLM Call 1 - Consolidated) ---

def get_period_portfolio_recommendation(row: pd.Series, portfolio: Dict[str, Any]) -> pd.Series:
    """
    Analyzes an aggregated period of news (one row) and generates a structured 
    portfolio recommendation using a single LLM call.
    """
    
    # 1. Extract Period-Level Context (from input file columns)
    period_summary = row.get('consolidated_news_summary', 'No summary available.')
    period_risk_score = row.get('consolidated_risk_score_for_period', 'N/A')
    period_justification = row.get('consolidated_risk_justification_for_period', 'No overall justification provided.')
    period_timestamp = row.get('period_end_timestamp', 'Unknown Time')

    # 2. Construct the detailed portfolio and risk context
    portfolio_context = f"--- Period-Level Risk Assessment (Ending {period_timestamp}) ---\n"
    
    # Add overall period risk
    portfolio_context += f"Overall Period Risk Score: {period_risk_score}\n"
    portfolio_context += f"Overall Period Justification: {period_justification}\n\n"
    
    # Add all 10 industry-specific scores and justifications
    for industry, details in portfolio.items():
        risk_col = clean_industry_name(industry, "_period_risk_score")
        justification_col = clean_industry_name(industry, "_period_justification")
        
        risk_score = row.get(risk_col, 'N/A')
        justification = row.get(justification_col, 'No justification provided.')

        portfolio_context += (
            f"- **{industry}** ({details['stock']} - {details['weight']}% weight)\n"
            f"  -> Period Risk Score (1-10): {risk_score}\n"
            f"  -> Justification: {justification[:100]}...\n" # Truncate for prompt length
        )
    
    # 3. Construct the full LLM prompt demanding JSON output
    prompt = f"""
You are a senior risk strategist advising a client with the following portfolio based on aggregated news for a specific period.
Your task is to provide a single, consolidated action recommendation and associated percentage change estimates.

**The Period Context (Consolidated News Summary):**
"{period_summary}"

**The Risk Assessment and Portfolio Context:**
{portfolio_context}

**Instruction:**
1. Determine the overall **recommendation_type** ('NO ACTION NEEDED', 'REBALANCE', or 'HEDGE/INCREASE').
2. Detail the specific **portfolio_action** (e.g., 'Reduce XOM by 5% and Increase AAPL by 5%') or 'None'.
3. Provide a concise **reasoning** for the decision based on all the risk data.
4. Estimate **portfolio_return_without_action** and **portfolio_return_with_action** (as a percentage, e.g., 0.5 or -1.2, not a string).
5. Calculate the **potential_gains** (the difference between the 'with action' and 'without action' returns).

You MUST return ONLY a valid JSON object matching the following structure.

Expected JSON Format:
{{
    "recommendation_type": "...",
    "portfolio_action": "...",
    "reasoning": "...",
    "portfolio_return_without_action": ...,
    "portfolio_return_with_action": ...,
    "potential_gains": ...
}}
"""
    
    # Define default result series for error handling
    error_result = pd.Series({
        'recommendation_type': 'ERROR', 
        'portfolio_action': 'Check Log', 
        'reasoning': f'LLM/Parsing error for period: {period_timestamp}',
        'portfolio_return_without_action': None,
        'portfolio_return_with_action': None,
        'potential_gains': None,
    })

    try:
        # 4. Call Ollama API
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"]
        result = clean_llm_json(content)
        
        # 5. Return structured result as a Series
        return pd.Series({
            'recommendation_type': result.get('recommendation_type', 'UNKNOWN'), 
            'portfolio_action': result.get('portfolio_action', 'None Specified'), 
            'reasoning': result.get('reasoning', 'No reasoning provided.'),
            'portfolio_return_without_action': result.get('portfolio_return_without_action'),
            'portfolio_return_with_action': result.get('portfolio_return_with_action'),
            'potential_gains': result.get('potential_gains'),
        })
    
    except Exception as e:
        print(f"❌ Error processing row (LLM/JSON): {e}")
        return error_result


# --- Main Execution ---

if __name__ == "__main__":
    
    print(f"🚀 Starting Consolidated Portfolio Action Analysis on aggregated file: {FILE_PATH}")
    
    # 🌟 STEP 1: Load the DataFrame from the specified CSV file
    try:
        df = pd.read_csv(FILE_PATH)
        
        # Check for essential columns from the aggregated file structure
        required_cols = ['consolidated_news_summary', 'consolidated_risk_score_for_period']
        if not all(col in df.columns for col in required_cols):
             raise ValueError(f"Required columns {required_cols} are missing from the CSV file.")

        print(f"✅ Successfully loaded {len(df)} period records from {FILE_PATH}")

    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
        
    # 🌟 STEP 2: STAGE 1 - Portfolio Action Recommendation (Single LLM Call)
    print("\n--- STAGE 1: Consolidated Portfolio Action Recommendation (LLM Call 1) ---")
    
    recommendation_results = df.apply(
        lambda row: get_period_portfolio_recommendation(row, portfolio=HYPOTHETICAL_PORTFOLIO), 
        axis=1
    )
    
    # Merge the results back into the original DataFrame
    df = pd.concat([df, recommendation_results], axis=1)

    print("✅ Stage 1 complete.")

    # 🌟 STEP 3: Display and Save the results
    print("\n--- FINAL CONSOLIDATED PORTFOLIO REPORT WITH ACTIONS ---")
    
    # Select columns for final display
    final_cols = [
        'period_end_timestamp', 
        'consolidated_risk_score_for_period',
        'consolidated_news_summary', 
        'recommendation_type', 
        'portfolio_action', 
        'reasoning',
        'portfolio_return_without_action',
        'portfolio_return_with_action',
        'potential_gains'
    ]

    print(df[final_cols].head())

    OUTPUT_FILE = "final_portfolio_actions_report.csv"
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Final results saved to {OUTPUT_FILE}")