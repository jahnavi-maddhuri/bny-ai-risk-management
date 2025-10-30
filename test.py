import pandas as pd
import ollama
import json
import re
import sys
from typing import Dict, Any

# --- Configuration and Constants ---

# The 10 industries whose risk scores are in the input CSV
PORTFOLIO_INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "Energy",
    "Manufacturing", "Transportation", "Retail",
    "Telecommunications", "Real Estate", "Agriculture"
]

# The model to use for all LLM calls
OLLAMA_MODEL = "gemma3:1b" 

# File path for reading the input data (must contain 'published_utc' and 'summary')
FILE_PATH = "data/jpm_risk_assessment_output.csv"

# --- Helper Functions ---

def clean_industry_name(industry_name: str, suffix: str) -> str:
    """Converts 'Real Estate' to 'Real_Estate_risk_score' format expected in the CSV."""
    return industry_name.replace(" ", "_").replace("&", "and").replace("/", "") + suffix

def clean_llm_json(content: str) -> Dict[str, Any]:
    """Extracts, cleans, and parses the JSON object from the LLM output."""
    json_match = re.search(r'\{[\s\S]*\}', content)
    if not json_match:
        raise ValueError("No valid JSON object found in model output.")
    
    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_match.group(0))
    return json.loads(json_str)

# --- Stage 1: Per-News Consolidation (LLM Call 1) ---

def get_consolidated_risk_score(row: pd.Series) -> pd.Series:
    """
    Analyzes an entire DataFrame row to generate a single consolidated risk score (1-10) 
    and justification based on individual industry scores.
    """
    news_summary = row.get('summary', 'No summary available.')
    
    # Construct the detailed risk context listing all 10 scores
    risk_summary_context = "--- Individual Industry Risk Scores (1-10, 10 is highest risk) ---\n"
    for industry in PORTFOLIO_INDUSTRIES:
        risk_col = clean_industry_name(industry, "_risk_score")
        risk_score = row.get(risk_col, 'N/A')
        score_str = str(risk_score) if pd.notna(risk_score) else "N/A"
        risk_summary_context += f"- **{industry}**: {score_str}\n"
    
    prompt = f"""
You are an expert risk assessment system. Review the news summary and the specific industry risk scores (1-10) below.
You must synthesize this information into a single, overall **Consolidated Risk Score (1-10)** for the total impact of this news, and provide a clear justification.

**The News Context:**
"{news_summary}"

**The Risk Scores:**
{risk_summary_context}

Return ONLY a valid JSON object matching the following structure.
{{
    "consolidated_risk_score": ...,
    "overall_justification": "..."
}}
"""
    error_result = pd.Series({
        'consolidated_risk_score': None, 
        'overall_justification': f'LLM/Parsing error for row with summary: {news_summary[:50]}...'
    })

    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        result = clean_llm_json(response["message"]["content"])
        
        score = result.get('consolidated_risk_score')
        return pd.Series({
            'consolidated_risk_score': int(score) if score is not None else None, 
            'overall_justification': result.get('overall_justification', 'No reasoning provided.'),
        })
    
    except Exception as e:
        print(f"❌ Error (Stage 1): {e} for news item: {news_summary[:30]}...")
        return error_result

# --- Stage 2: Time Aggregation (LLM Call 2) ---

def synthesize_period_risk(group_df: pd.DataFrame) -> pd.Series:
    """
    Aggregates a time-based group of news items and uses an LLM to create 
    a single summary, score, and justification for the entire period.
    """
    if group_df.empty:
        return pd.Series({'time_summary': '', 'period_risk_type': 'NO NEWS', 
                          'period_risk_score': None, 'period_justification': ''})
    
    # 1. Prepare aggregated inputs
    
    # Concatenate all summaries and justifications
    all_summaries = group_df['summary'].str.cat(sep='\n---NEXT NEWS ITEM---\n')
    
    # List all consolidated risk scores generated in Stage 1
    score_list = group_df['consolidated_risk_score'].dropna().tolist()
    
    # Determine the time-period label
    period_end_time = group_df.index[-1]
    
    # 2. Construct the LLM prompt for period synthesis
    prompt = f"""
You are a time-series risk synthesis engine. Analyze the following collection of news items and their individually assessed risk scores (1-10) over a single time period ending at {period_end_time}.
Your goal is to provide a single, **Period-Level** risk assessment.

**Period Input Data:**
Individual Consolidated Risk Scores: {score_list}
---
Consolidated News Summary:
{all_summaries}
---

**Instruction:**
1. Determine the overall **Period-Level Risk Score** (1-10).
2. Write a concise, consolidated **News Summary** for this period.
3. Provide the **Justification** for the final Period-Level Risk Score.

Return ONLY a valid JSON object matching the following structure.
- 'period_risk_type' must be 'LOW RISK', 'MODERATE RISK', or 'HIGH RISK'.
- 'period_risk_score' must be an integer from 1 to 10.
- 'time_summary' must be a single paragraph synthesizing the key events.

Expected JSON Format:
{{
    "period_risk_type": "...",
    "period_risk_score": ...,
    "period_justification": "...",
    "time_summary": "..."
}}
"""
    error_result = pd.Series({
        'time_summary': f"Error synthesizing {len(group_df)} items.",
        'period_risk_type': 'ERROR', 
        'period_risk_score': None, 
        'period_justification': f'LLM/Parsing error during period aggregation.'
    })

    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        result = clean_llm_json(response["message"]["content"])
        
        score = result.get('period_risk_score')
        
        return pd.Series({
            'time_summary': result.get('time_summary', all_summaries),
            'period_risk_type': result.get('period_risk_type', 'UNKNOWN'),
            'period_risk_score': int(score) if score is not None else None, 
            'period_justification': result.get('period_justification', 'No justification provided.'),
        })
    
    except Exception as e:
        print(f"❌ Error (Stage 2): {e} for period ending: {period_end_time}")
        return error_result

# --- Main Execution ---

if __name__ == "__main__":
    
    # 🌟 STEP 1: Get user input for N hours
    try:
        N_HOURS = 6
        if N_HOURS <= 0:
             raise ValueError("Interval must be a positive integer.")
    except ValueError as e:
        print(f"Invalid input: {e}. Exiting.")
        sys.exit(1)
        
    print(f"\nSetting aggregation interval to {N_HOURS} hours.")
    
    # 🌟 STEP 2: Load and Prepare Data
    try:
        df = pd.read_csv(FILE_PATH)
        df = df.dropna(subset=['published_utc', 'summary']).copy()
        
        # Convert timestamp and set as index
        df['published_utc'] = pd.to_datetime(df['published_utc'])
        df = df.sort_values('published_utc')
        
        print(f"✅ Successfully loaded {len(df)} news items from {FILE_PATH}")

    except FileNotFoundError:
        print(f"❌ Error: The file {FILE_PATH} was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading or validating data: {e}")
        sys.exit(1)
        
    # 🌟 STEP 3: Stage 1 - Calculate Per-News Consolidated Risk (if not already done)
    if 'consolidated_risk_score' not in df.columns:
        print("\n--- STAGE 1: Calculating Per-News Consolidated Risk Scores (LLM Call 1) ---")
        
        # Apply the consolidation function to get per-news scores
        consolidation_results = df.apply(get_consolidated_risk_score, axis=1)
        df = pd.concat([df, consolidation_results], axis=1)
        
        print(f"✅ Stage 1 complete. Sample consolidated score: {df['consolidated_risk_score'].head(2).tolist()}")
    else:
        print("\n--- STAGE 1: 'consolidated_risk_score' already exists. Skipping first LLM call. ---")
    
    df = df.set_index('published_utc')
    
    # 🌟 STEP 4: Stage 2 - Time-Series Aggregation (LLM Call 2)
    print(f"\n--- STAGE 2: Aggregating into {N_HOURS}-hour windows (LLM Call 2) ---")
    
    # Resample the DataFrame using the user-defined interval
    resampling_interval = f'{N_HOURS}H'
    
    # Apply the synthesis function to each time group
    aggregated_df = df.resample(resampling_interval).apply(synthesize_period_risk)
    
    # Clean up the resulting DataFrame
    aggregated_df = aggregated_df.dropna(subset=['period_risk_score']).reset_index()
    
    # Rename columns for final output
    final_output_cols = {
        'published_utc': 'period_end_timestamp',
        'time_summary': 'consolidated_news_summary',
        'period_risk_score': 'consolidated_risk_score_for_period',
        'period_justification': 'consolidated_risk_justification_for_period',
    }
    aggregated_df = aggregated_df.rename(columns=final_output_cols)
    
    # Select final columns in the requested order
    final_df = aggregated_df[[
        'period_end_timestamp', 
        'consolidated_news_summary', 
        'consolidated_risk_score_for_period', 
        'consolidated_risk_justification_for_period'
    ]]

    # 🌟 STEP 5: Display and Save Results
    print("\n--- FINAL TIME-AGGREGATED RISK REPORT ---")
    print(f"Results for {N_HOURS}-hour intervals:")
    print(final_df)

    OUTPUT_FILE = f"portfolio_risk_aggregated_{N_HOURS}hr.csv"
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Final aggregated results saved to {OUTPUT_FILE}")
