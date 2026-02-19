import pandas as pd
import requests
import json
import time
from typing import Dict, Any
import yfinance as yf
from datetime import timedelta

# ==============================
# CONFIG
# ==============================
OLLAMA_MODEL = "qwen2.5:3b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"
INPUT_CSV = "OUTPUT_Largest Negative Changes US DXY - Sheet1.csv"
OUTPUT_CSV = "FULL_OUTPUT_Largest Negative Changes US DXY - Sheet1.csv"
SLEEP_BETWEEN_CALLS = 0.5


# ==============================
# PROMPT TEMPLATE
# ==============================
PROMPT_TEMPLATE = """You are a precise macro FX strategist specializing in the US Dollar Index (DXY). Analyze the following news article and determine:

1. Does this article describe an event that would impact the DXY (US Dollar Index)?
2. If yes, which of the 25 major FX events does it represent?
3. What is the anticipated impact on the DXY?

Note: DXY reflects the value of the US dollar against a basket of major currencies (EUR, JPY, GBP, CAD, SEK, CHF). Focus on broad USD strength or weakness, not a single bilateral pair.

## ARTICLE TO ANALYZE:

Content: {content}
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

    if "Date" not in df.columns:
        raise ValueError('CSV must contain a column named "Date"')

    # Prepare classification output columns
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

    # Market feature columns
    market_columns = [
        "vol_lag_1d", "vol_lag_7d", "vol_lag_30d", "vol_lag_365d",
        "mom_lag_1d", "mom_lag_7d", "mom_lag_30d", "mom_lag_365d"
    ]

    for col in market_columns:
        if col not in df.columns:
            df[col] = None

    # Convert dates
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y", errors="coerce")

    # Download DXY once
    min_date = df["Date"].min() - timedelta(days=400)
    max_date = df["Date"].max() + timedelta(days=5)



    dxy = yf.download("DX-Y.NYB", start=min_date, end=max_date, progress=False)

    if dxy.empty:
        raise ValueError("Failed to download DXY data")

    dxy = dxy.sort_index()

    for idx, row in df.iterrows():
        content = str(row["full_text"])
        event_date = row["Date"]

        if pd.isna(event_date):
            print(f"Invalid date at row {idx}")
            continue

        try:
            px_event = dxy.loc[:event_date].iloc[-1]["Close"]

            def get_last_available(dxy, target_date):
                """
                Returns the last row in dxy on or before target_date.
                Returns None if no data exists.
                """
                subset = dxy[dxy.index <= target_date]
                if subset.empty:
                    return None
                return subset.iloc[-1]

            def get_lag_metrics(dxy, event_date, days_lag):
                lag_date = event_date - pd.Timedelta(days=days_lag)
                row = get_last_available(dxy, lag_date)
                print(row)
                if row is None:
                    return None, None
                px_lag = row["Close"]
                vol_lag = row["Volume"]
                return vol_lag, None if pd.isna(px_lag) else (dxy.loc[:event_date].iloc[-1]["Close"] / px_lag - 1)


            vol1, mom1 = get_lag_metrics(dxy, event_date, 1)
            vol7, mom7 = get_lag_metrics(dxy, event_date, 7)
            vol30, mom30 = get_lag_metrics(dxy, event_date, 30)
            vol365, mom365 = get_lag_metrics(dxy, event_date, 365)

            df.at[idx, "vol_lag_1d"] = vol1
            df.at[idx, "vol_lag_7d"] = vol7
            df.at[idx, "vol_lag_30d"] = vol30
            df.at[idx, "vol_lag_365d"] = vol365

            df.at[idx, "mom_lag_1d"] = mom1
            df.at[idx, "mom_lag_7d"] = mom7
            df.at[idx, "mom_lag_30d"] = mom30
            df.at[idx, "mom_lag_365d"] = mom365

        except Exception as e:
            print(f"Market data error at row {idx}: {e}")
            continue

        # Inject market context into prompt
        market_context = f"""

## DXY MARKET CONTEXT:

DXY Close on Event Date: {px_event}

Volume:
- 1d lag: {vol1}
- 7d lag: {vol7}
- 30d lag: {vol30}
- 365d lag: {vol365}

Momentum (percent return to event date):
- 1d: {mom1}
- 7d: {mom7}
- 30d: {mom30}
- 365d: {mom365}
"""

        prompt = PROMPT_TEMPLATE.format(content=content) + market_context

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
