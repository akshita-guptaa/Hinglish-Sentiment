"""
Evaluation script for the Hinglish Normalizer + Sentiment pipeline.
"""
import os
import sys

import pandas as pd
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from processor import (  # noqa: E402
    load_config,
    load_prompt_template,
    init_gemini_client,
    process_dataframe,
)

TEST_PATH = os.path.join(ROOT, "eval", "test_set.csv")
RESULT_PATH = os.path.join(ROOT, "eval", "eval_results.csv")

def to_bool(value):
    """Convert True/False, 'true'/'false' strings into a real boolean."""
    return str(value).strip().lower() == "true"

def main():
    load_dotenv(os.path.join(ROOT, ".env"))
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("[ERROR] GEMINI_API_KEY not found in .env")
 
    os.chdir(ROOT)
    config = load_config("config.yaml")
    prompt_template = load_prompt_template(config["prompt_path"])
    client = init_gemini_client(api_key)

    test_df = pd.read_csv(TEST_PATH)
    print(f"Running pipeline on {len(test_df)} labeled sentences...\n")

    out = process_dataframe(
        df=test_df[["text"]],
        client=client,
        config=config,
        prompt_template=prompt_template,
        progress_callback=lambda cur, total, pct: print(
            f"--> Batch {cur}/{total} ({pct:.0%})"
        ),
    ).reset_index(drop=True)

    if len(out) != len(test_df):
        sys.exit("[ERROR] Output rows do not match test set rows.")

    # Put expected labels and predictions side by side (same row order)
    result = test_df.copy()
    result["pred_sentiment"] = out["sentiment"].astype(str).str.lower()
    result["pred_sarcasm"] = out["sarcasm"].apply(to_bool)
    result["expected_sarcasm"] = result["expected_sarcasm"].apply(to_bool)
    result["confidence"] = out["confidence"]

    # Rows where the API failed are not counted as right or wrong
    failed_mask = result["pred_sentiment"].isin(["error", "skipped"])
    valid = result[~failed_mask]
    failed = int(failed_mask.sum())
    if valid.empty:
        sys.exit("[ERROR] No valid predictions. Check your API key and quota.")

    sent_ok = valid["pred_sentiment"] == valid["expected_sentiment"]
    sarc_ok = valid["pred_sarcasm"] == valid["expected_sarcasm"]

    print("\n" + "=" * 55)
    print(f"Sentences evaluated   : {len(valid)} (API failures skipped: {failed})")
    print(f"Sentiment accuracy    : {sent_ok.mean():.1%}  ({sent_ok.sum()}/{len(valid)})")
    print(f"Sarcasm accuracy      : {sarc_ok.mean():.1%}  ({sarc_ok.sum()}/{len(valid)})")

    sarcastic = valid[valid["expected_sarcasm"]]
    if not sarcastic.empty:
        caught = (sarcastic["pred_sarcasm"]).sum()
        print(f"Sarcasm recall        : {caught}/{len(sarcastic)} sarcastic sentences caught")
    print("=" * 55)

    print("\nSentiment confusion table (rows = expected, columns = predicted):")
    print(pd.crosstab(valid["expected_sentiment"], valid["pred_sentiment"]))

    wrong = valid[~(sent_ok & sarc_ok)]
    print(f"\nMismatches ({len(wrong)}):")
    if wrong.empty:
        print("None")
    else:
        print(
            wrong[
                ["text", "expected_sentiment", "pred_sentiment",
                 "expected_sarcasm", "pred_sarcasm"]
            ].to_string(index=False)
        )

    result.to_csv(RESULT_PATH, index=False, encoding="utf-8")
    print(f"\nFull results saved to {os.path.relpath(RESULT_PATH, ROOT)}")


if __name__ == "__main__":
    main()