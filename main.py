"""
Hinglish Normalizer + Sentiment Pipeline - CLI Entry Point
----------------------------------------------------------
Reads Hinglish (Hindi-English code-mixed) texts from a CSV, processes them
in batches using Google Gemini via the core processor module, validates
the output, and writes the enriched results to a CSV.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Import reusable engine functions from processor.py
from processor import (
    load_config,
    load_prompt_template,
    init_gemini_client,
    process_dataframe,
    process_batch_with_retry,
    call_gemini_api,
    extract_and_parse_json,
    validate_batch_response,
)


def main():
    print("=" * 65)
    print(" Hinglish Normalizer + Sentiment Analysis Pipeline ")
    print("=" * 65)

    # 1. Load Environment Variables (.env)
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found in environment or .env file.")
        print("Please create a .env file and set GEMINI_API_KEY='your_api_key_here'")
        sys.exit(1)

    # 2. Load Configuration and Prompt Template
    try:
        config = load_config("config.yaml")
        prompt_template = load_prompt_template(config["prompt_path"])
        client = init_gemini_client(api_key)
    except Exception as e:
        print(f"[Initialization Error] {e}")
        sys.exit(1)

    print(f"Model: {config['model_name']} | Batch Size: {config['batch_size']} | Temperature: {config['temperature']}")
    print(f"Input: {config['input_path']}  -->  Output: {config['output_path']}\n")

    # 3. Read Input CSV
    if not os.path.exists(config["input_path"]):
        print(f"[ERROR] Input file not found: {config['input_path']}")
        sys.exit(1)

    try:
        df_input = pd.read_csv(config["input_path"])
    except Exception as e:
        print(f"[ERROR] Failed to read input CSV: {e}")
        sys.exit(1)

    # Progress logger callback for batch tracking
    def log_progress(current_batch, total_batches, percentage):
        print(f"--> Processing Batch {current_batch}/{total_batches} ({percentage:.0%})...")

    # 4. Process all rows via core engine
    df_output = process_dataframe(
        df=df_input,
        client=client,
        config=config,
        prompt_template=prompt_template,
        progress_callback=log_progress
    )

    # 5. Export to Output CSV
    output_dir = os.path.dirname(config["output_path"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df_output.to_csv(config["output_path"], index=False, encoding="utf-8")

    print("\n" + "=" * 65)
    print(" Processing Complete! ")
    print("=" * 65)
    print(f"Results successfully saved to: {config['output_path']}")
    print(f"Total processed rows: {len(df_output)}")
    print(f"Columns: {', '.join(df_output.columns)}")

    # Summary Statistics
    if not df_output.empty:
        print("\nSummary of Sentiments:")
        valid_sentiments = df_output[df_output["sentiment"].isin(["positive", "negative", "neutral"])]
        if not valid_sentiments.empty:
            print(valid_sentiments["sentiment"].value_counts().to_string())

        error_count = (df_output["sentiment"] == "ERROR").sum()
        if error_count > 0:
            print(f"\n[Notice] {error_count} row(s) encountered service errors and marked as ERROR.")

        print(f"Sarcastic items detected: {df_output['sarcasm'].sum()} / {len(df_output)}")


if __name__ == "__main__":
    main()
