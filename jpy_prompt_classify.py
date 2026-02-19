import pandas as pd
import requests
import json
import time
from typing import Dict, Any

# ==============================
# CONFIG
# ==============================
OLLAMA_MODEL = "qwen2.5:3b-instruct"   # change to your local ollama model
OLLAMA_URL = "http://localhost:11434/api/generate"
INPUT_CSV = "OUTPUT_Largest Negative Changes US DXY - Sheet1.csv"
OUTPUT_CSV = "FULL_OUTPUT_Largest Negative Changes US DXY - Sheet1.csv"
SLEEP_BETWEEN_CALLS = 0.5  # avoid overwhelming local server


# ==============================
# PROMPT TEMPLATE
# ==============================
PROMPT_TEMPLATE = """You are a precise macro FX strategist specializing in the US Dollar Index (DXY). Analyze the following news article and determine:

1. Does this article describe an event that would impact the DXY (US Dollar Index)?
2. If yes, which of the 25 major FX events does it represent?
3. What is the anticipated impact on the DXY?

Note: DXY reflects the value of the US dollar against a basket of major currencies (EUR, JPY, GBP, CAD, SEK, CHF). Focus on broad USD strength or weakness, not a single bilateral pair.

## THE 25 MAJOR FX EVENTS:

*US Monetary Policy:*
1. Fed Rate Hike
2. Fed Rate Cut
3. Hawkish Fed Guidance
4. Dovish Fed Guidance
5. Fed QE Announcement
6. Fed QT/Tapering

*Foreign Monetary Policy (Major DXY Components):*
7. ECB Rate Hike
8. ECB Rate Cut
9. BoJ Policy Tightening
10. BoJ Policy Easing
11. BoE Policy Tightening
12. BoE Policy Easing

*US Economic Data:*
13. US Inflation Above Expected
14. US Inflation Below Expected
15. Strong US Employment (NFP)
16. Weak US Employment
17. US GDP Growth Beats
18. US GDP Contraction/Recession

*Foreign Economic Data (Major DXY Components):*
19. Strong Eurozone Data
20. Weak Eurozone Data
21. Strong Japan/UK Data
22. Weak Japan/UK Data

*Intervention & Market:*
23. Coordinated FX Intervention
24. Global Risk-Off Event/Crisis
25. Global Risk-On/Equity Rally

## CLASSIFICATION RULES:

DO NOT CLASSIFY if the article is:
• Opinion pieces or analyst forecasts without new data  
• Historical analysis without new developments  
• Generic market commentary  
• Technical analysis or price predictions  
• Non-official sources speculating about future events  

DO CLASSIFY if the article reports:
• Actual data releases (CPI, NFP, GDP, etc.)  
• Official central bank announcements or speeches  
• Confirmed policy changes  
• Major geopolitical events affecting global markets  
• Confirmed intervention or systemic crisis events  

## IMPACT ASSESSMENT:

For DXY direction:
• Events strengthening USD relative to the basket → "UP"
• Events weakening USD relative to the basket → "DOWN"
• Mixed or offsetting cross-currency effects → "NEUTRAL"

For magnitude:
• HIGH: Major global policy shift or systemic shock  
• MEDIUM: Significant data surprise or strong guidance shift  
• LOW: Minor data surprise or limited macro relevance  

You MUST provide clear macro reasoning for every field in the output, especially:
• Why the article is or is not FX relevant  
• Why the specific event number was chosen  
• Why the direction and magnitude were assigned  

## ARTICLE TO ANALYZE:

Content: {content}

## REQUIRED OUTPUT FORMAT (JSON):
Return ONLY valid JSON matching this schema:

{{
  "is_fx_relevant": true/false,
  "event_number": 1-25 or null,
  "event_name": "Event name" or null,
  "confidence": "high"/"medium"/"low",
  "dxy_impact": "UP"/"DOWN"/"NEUTRAL" or null,
  "magnitude": "HIGH"/"MEDIUM"/"LOW" or null,
  "reasoning": "Detailed macro explanation covering classification logic, USD transmission channel, and magnitude assessment"
}}

## EXAMPLES

### Example 1
Article summary: US CPI rises 0.6% month-over-month versus 0.3% expected.

Expected JSON output:
{{
  "is_fx_relevant": true,
  "event_number": 13,
  "event_name": "US Inflation Above Expected",
  "confidence": "high",
  "dxy_impact": "UP",
  "magnitude": "MEDIUM",
  "reasoning": "Official CPI release with upside surprise. Higher inflation increases probability of tighter Fed policy, raising US yields and strengthening USD versus the DXY basket."
}}

### Example 2
Article summary: ECB unexpectedly cuts rates by 50 basis points and signals further easing.

Expected JSON output:
{{
  "is_fx_relevant": true,
  "event_number": 8,
  "event_name": "ECB Rate Cut",
  "confidence": "high",
  "dxy_impact": "UP",
  "magnitude": "HIGH",
  "reasoning": "Official ECB rate cut. Lower Eurozone yields weaken EUR, the largest DXY component, mechanically pushing DXY higher."
}}

### Example 3
Article summary: Analyst argues that the dollar may weaken later this year.

Expected JSON output:
{{
  "is_fx_relevant": false,
  "event_number": null,
  "event_name": null,
  "confidence": "high",
  "dxy_impact": null,
  "magnitude": null,
  "reasoning": "Opinion-based forecast without new data or policy action. Does not meet classification criteria."
}}
"""




# ==============================
# OLLAMA CALL
# ==============================
def call_ollama(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    top_p: float = 0.95,
    stop: list = None
) -> Dict[str, Any]:
    """
    Calls the local Ollama model with configurable parameters.
    Returns a parsed JSON dict from the model's output.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }

    if stop:
        payload["stop"] = stop

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    raw = response.json().get("response", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # fallback cleanup if model adds stray text
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(raw[start:end])
        raise ValueError("Model did not return valid JSON")



# ==============================
# MAIN PIPELINE
# ==============================
def main():
    df = pd.read_csv(INPUT_CSV)

    if "full_text" not in df.columns:
        raise ValueError('CSV must contain a column named "full_text"')

    # Prepare output columns
    output_columns = [
        "is_fx_relevant",
        "event_number",
        "event_name",
        "confidence",
        "dxy_impact",
        "magnitude",
        "reasoning"
    ]

    for col in output_columns:
        if col not in df.columns:
            df[col] = None

    for idx, row in df.iterrows():
        content = str(row["full_text"])

        prompt = PROMPT_TEMPLATE.format(content=content)

        try:
            result = call_ollama(prompt)

            for col in output_columns:
                df.at[idx, col] = result.get(col)

            print(f"Processed row {idx}")

        except Exception as e:
            print(f"Error at row {idx}: {e}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
