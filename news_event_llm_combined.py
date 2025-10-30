import pandas as pd 
import ollama
import json
import re
from typing import Dict, Any, Union

# --- Configuration (Copied from previous context for completeness) ---

# The 10 industries whose risk scores are in the input CSV
PORTFOLIO_INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "Energy",
    "Manufacturing", "Transportation", "Retail",
    "Telecommunications", "Real Estate", "Agriculture"
]

# The model to use (as per your previous request)
OLLAMA_MODEL = "gemma3:1b" 

# File path for reading the input data (assumed to be the source of rows)
FILE_PATH = "data/jpm_risk_assessment_output.csv"

def clean_industry_name(industry_name: str, suffix: str) -> str:
    """Converts 'Real Estate' to 'Real_Estate_risk_score' format expected in the CSV."""
    return industry_name.replace(" ", "_").replace("&", "and").replace("/", "") + suffix

# --- Core Function to Generate Consolidated Risk Score ---

def get_consolidated_risk_score(row: pd.Series) -> pd.Series:
    """
    Analyzes an entire DataFrame row to generate a single consolidated risk score (1-10) 
    and overall justification based on individual industry scores.
    """
    # Safely get the news summary
    news_summary = row.get('summary', 'No summary available.')
    
    # 1. Construct the detailed risk context listing all 10 scores
    risk_summary_context = "--- Individual Industry Risk Scores (1-10, 10 is highest risk) ---\n"
    
    for industry in PORTFOLIO_INDUSTRIES:
        # Get risk score column dynamically
        risk_col = clean_industry_name(industry, "_risk_score")
        
        # Safely extract data
        risk_score = row.get(risk_col, 'N/A')

        score_str = str(risk_score) if pd.notna(risk_score) else "N/A (Missing Data)"
        
        risk_summary_context += (
            f"- **{industry}**: {score_str}\n"
        )
    
    # 2. Construct the full LLM prompt demanding JSON output
    prompt = f"""
You are an expert risk assessment system. Your task is to review the news summary and the specific industry risk scores (1-10) below.
You must synthesize this information into a single, overall **Consolidated Risk Score (1-10)** for the entire portfolio impact, and provide a clear justification.\
Keep in mind that if you provide industry risk scores of particular values, the overall risk score should lie within a similar range and should reflect the combination of industry risks, and should not \
be significantly higher. Do not add risks across industries, but ideally average them out or use a similar method, the consolidated risk score can not be higher than all the other individual industry risks.

**The News Context:**
"{news_summary}"

**The Risk Scores:**
{risk_summary_context}

**Instruction:**
Return ONLY a valid JSON object matching the following structure.
- 'consolidated_risk_score' must be an integer from 1 to 10.
- 'overall_justification' must be a concise explanation of the final score based on the distribution of the individual scores.

Expected JSON Format:
{{
    "consolidated_risk_score": ...,
    "overall_justification": "..."
}}
"""
    
    # Define default result series for error handling
    error_result = pd.Series({
        'consolidated_risk_score': None, 
        'overall_justification': f'LLM/Parsing error for row with summary: {news_summary[:50]}...'
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
        
        # 4. Return structured result as a Series, ensuring the score is an integer
        score = result.get('consolidated_risk_score')
        
        return pd.Series({
            'consolidated_risk_score': int(score) if score is not None else None, 
            'overall_justification': result.get('overall_justification', 'No reasoning provided.'),
        })
    
    except Exception as e:
        print(f"❌ Error processing row (LLM/JSON): {e}")
        return error_result


# --- 5. Main Execution Example ---

if __name__ == "__main__":
    
    # 🌟 STEP 1: Load the DataFrame (Assume it exists)
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"✅ Successfully loaded data from {FILE_PATH}")
        print(f"Data shape: {df.shape}")
        
    except FileNotFoundError:
        print(f"❌ Error: The file {FILE_PATH} was not found.")
        print("Please ensure the file exists at the specified path.")
        exit()
        
    # Print a sample of the input data for verification
    sample_cols = ['summary'] + [clean_industry_name(i, "_risk_score") for i in PORTFOLIO_INDUSTRIES]
    print("\n--- Sample of Input Data (Summary and Risk Scores) ---")
    print(df[sample_cols].head(2))
    
    # 🌟 STEP 2: Use df.apply(..., axis=1) to process each row
    print("\n--- Applying Consolidated Risk Analysis (Calling LLM on each row) ---\n")
    
    # The result of apply will be a DataFrame containing the two new columns
    consolidated_results = df.apply(
        lambda row: get_consolidated_risk_score(row), 
        axis=1
    )
    
    # Merge the results back into the original DataFrame
    df = pd.concat([df, consolidated_results], axis=1)

    # 🌟 STEP 3: Display and Save the results
    
    output_cols = ['summary', 'consolidated_risk_score', 'overall_justification']
    print("\n--- Final DataFrame with Consolidated Risk Scores (First 5 Rows) ---")
    print(df[output_cols].head())

    OUTPUT_FILE = "portfolio_risk_consolidated_time.csv"
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Results saved to {OUTPUT_FILE}")
