import ollama
import json
import re
import pandas as pd

def analyze_news_risk(news_text, industries_list):
    """
    Analyze a single news item and return a Series where each industry's
    risk_score and justification are separate columns.

    This version safely escapes the news_text and cleans LLM output
    to prevent invalid control character errors.
    """

    if industries_list is None:
        industries_to_analyze = [
            "Technology", "Finance", "Healthcare", "Energy",
            "Manufacturing", "Transportation", "Retail",
            "Telecommunications", "Real Estate", "Agriculture"
        ]
    else:
        industries_to_analyze = industries_list

    # Escape the news_text so JSON remains valid
    safe_news_text = json.dumps(news_text)

    industries_list_for_prompt = "\n".join(
        [f"{i+1}. {industry}" for i, industry in enumerate(industries_to_analyze)]
    )

    prompt_template = f"""
You are an expert risk analyst. Assess the risk impact of the following news on each of the listed areas.
You must only consider these areas, and you must return a risk score and justification for *each one*.

Return ONLY valid JSON as shown below, including all areas from the list. Escape all special characters inside strings.

Expected JSON Format:
{{
    "risk_analysis": [
        {{
            "industry": "...",
            "risk_score": ...,
            "justification": "..."
        }}
    ]
}}

News:
{safe_news_text}

Areas to Analyze:
{industries_list_for_prompt}
"""

    try:
        # Call Ollama API
        response = ollama.chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": prompt_template}]
        )

        content = response["message"]["content"]

        # Extract JSON substring
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            raise ValueError("No valid JSON detected in model output")

        json_str = json_match.group(0)

        # Clean unescaped control characters
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)

        # Parse JSON
        result = json.loads(json_str)

        risk_analysis = result.get("risk_analysis", [])
        row_data = {"news_item": news_text}

        for item in risk_analysis:
            industry = item.get("industry", "").strip()
            score = item.get("risk_score", None)
            justification = item.get("justification", "")

            if not industry:
                continue

            # Clean column names for pandas
            score_col = f"{industry}_risk_score".replace(" ", "_").replace("&", "and").replace("/", "")
            just_col = f"{industry}_justification".replace(" ", "_").replace("&", "and").replace("/", "")

            row_data[score_col] = score
            row_data[just_col] = justification

        # Ensure all expected industries exist even if missing from LLM output
        for industry in industries_to_analyze:
            score_col = f"{industry}_risk_score".replace(" ", "_").replace("&", "and").replace("/", "")
            just_col = f"{industry}_justification".replace(" ", "_").replace("&", "and").replace("/", "")

            if score_col not in row_data:
                row_data[score_col] = None
            if just_col not in row_data:
                row_data[just_col] = None

        return pd.Series(row_data)

    except Exception as e:
        print(f"❌ Error analyzing news item: {e}")
        return pd.Series({"news_item": news_text})


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    df=pd.read_csv("data/jpm_press_releases.csv")

    portfolio_industries = [
            "Technology", "Finance", "Healthcare", "Energy",
            "Manufacturing", "Transportation", "Retail",
            "Telecommunications", "Real Estate", "Agriculture"
        ]

    risk_areas = [
        "Counterparty Risk",
        "Market Risk",
        "Operational Risk",
        "Liquidity Risk",
        "Regulatory/Compliance Risk",
        "Reputational Risk",
        "Strategic Risk",
        "Cybersecurity Risk",
        "Environmental Risk",
        "Geopolitical Risk"
    ]

    # Apply the function across the DataFrame
    expanded_df = df.apply(lambda x: analyze_news_risk(x["summary"], industries_list=portfolio_industries), axis=1)

    # Merge results back
    final_df = pd.concat([df, expanded_df.drop(columns=["news_item"])], axis=1)

    print("\n--- Full Industry-Wide Risk DataFrame ---\n")
    print(final_df.head())

    final_df.to_csv("data/jpm_risk_assessment_output.csv", index=False)
