
import ollama
import json
import re
import pandas as pd

def analyze_news_risk(news_text, industries_list):
    if industries_list is None:
        industries_to_analyze = [
            "Technology", "Finance", "Healthcare", "Energy",
            "Manufacturing", "Transportation", "Retail",
            "Telecommunications", "Real Estate", "Agriculture"
        ]
    else:
        industries_to_analyze = industries_list

    industries_list_for_prompt = "\n".join(
        [f"{i+1}. {industry}" for i, industry in enumerate(industries_to_analyze)]
    )

    prompt_template = f"""
    You are an expert risk analyst. Assess the risk impact of the following news on each of the listed areas.
    You must only consider these areas, and you must return a risk score and justification for *each one*.

    Return ONLY valid JSON as shown below, including all areas from the list:

    Expected JSON Format:
    {{
        "risk_analysis": [
            {{
                "industry": "Counterparty Risk",
                "risk_score": 7,
                "justification": "..."
            }},
            {{
                "industry": "Market Risk",
                "risk_score": 5,
                "justification": "..."
            }},
            ...
        ]
    }}
    News: {news_text}

    Industries:
    {industries_list_for_prompt}
    """

    try:
        # ✅ Correct API call (no format="json")
        response = ollama.chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": prompt_template}]
        )

        content = response["message"]["content"]

        # ✅ Extract JSON substring
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            raise ValueError("No valid JSON detected in model output")

        json_str = json_match.group(0)
        result = json.loads(json_str)

        risk_analysis = result.get("risk_analysis", [])
        row_data = {"news_item": news_text}

        for item in risk_analysis:
            industry = item.get("industry", "").strip()
            score = item.get("risk_score", None)
            justification = item.get("justification", "")

            if not industry:
                continue

            score_col = f"{industry}_risk_score".replace(" ", "_").replace("&", "and")
            just_col = f"{industry}_justification".replace(" ", "_").replace("&", "and")

            row_data[score_col] = score
            row_data[just_col] = justification

        return pd.Series(row_data)

    except Exception as e:
        print(f"❌ Error analyzing news item: {e}")
        return pd.Series({"news_item": news_text})


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    df = pd.DataFrame({
        "news_text": [
            f"A major new geopolitical conflict has erupted in a key maritime strait, halting all shipping traffic. Global oil prices have spiked 30% overnight.",
            "A ransomware attack has disrupted the operations of several hospitals, affecting patient data and emergency services."
        ]
    })

    top_10_industries = [
        "Technology (IT, Software, Hardware)",
        "Healthcare (Services, Equipment)",
        "Financial Services (Banking, Insurance)",
        "Energy (Oil & Gas, Renewables)",
        "Consumer Goods & Retail (incl. E-commerce)",
        "Automotive (Manufacturing, Sales)",
        "Telecommunications",
        "Pharmaceuticals & Biotechnology",
        "Construction & Real Estate",
        "Agriculture & Food Industry"
    ]
    risk_areas=["Counterparty Risk",
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
    expanded_df = df.apply(lambda x: analyze_news_risk(x["news_text"], industries_list=top_10_industries), axis=1)
    final_df = pd.concat([df, expanded_df.drop(columns=["news_item"])], axis=1)

    print("\n--- Full Industry-Wide Risk DataFrame ---\n")
    print(final_df.head())
