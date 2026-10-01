"""
Hinglish Normalizer + Sentiment Analysis - Core Processing Engine
----------------------------------------------------------------
This module provides reusable functions for loading configurations,
communicating with Google Gemini, validating structured JSON responses,
handling transient 503/429 errors with exponential backoff, failing over
to backup models, and batch/single text processing.
"""

import os
import sys
import json
import time
import re
import yaml
import pandas as pd
from dotenv import load_dotenv


class AIServiceBusyError(Exception):
    """Raised when the AI service is unavailable (503/429) across all retries and fallback models."""
    pass


# ==============================================================================
# 1. SDK IMPORT & CLIENT SETUP (Supports both modern google-genai and legacy)
# ==============================================================================
USE_MODERN_SDK = False
try:
    from google import genai
    from google.genai import types
    USE_MODERN_SDK = True
except ImportError:
    try:
        import google.generativeai as legacy_genai
        USE_MODERN_SDK = False
    except ImportError:
        pass


def load_config(config_path="config.yaml"):
    """
    Loads configuration settings from config.yaml.
    Provides default fallbacks in case any key is missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return {
        "model_name": config.get("model_name", "gemini-3.8-flash"),
        "fallback_models": config.get("fallback_models", ["gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest"]),
        "temperature": float(config.get("temperature", 0.2)),
        "batch_size": int(config.get("batch_size", 10)),
        "retry_limit": int(config.get("retry_limit", 1)),
        "input_path": config.get("input_path", "data/input.csv"),
        "output_path": config.get("output_path", "output/results.csv"),
        "prompt_path": config.get("prompt_path", "prompts/normalize.txt"),
    }


def load_prompt_template(prompt_path="prompts/normalize.txt"):
    """
    Reads the base prompt template containing guidelines and few-shot examples.
    """
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template file not found at: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def init_gemini_client(api_key=None):
    """
    Initializes and returns the Gemini client using the provided or environment API key.
    """
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key.strip() == "" or "your_gemini_api_key" in api_key:
        raise ValueError(
            "Invalid or missing GEMINI_API_KEY. "
            "Please configure your API key in .env or provide it directly."
        )

    if USE_MODERN_SDK:
        return genai.Client(api_key=api_key)
    else:
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=api_key)
            return legacy_genai
        except ImportError:
            raise ImportError("Neither google-genai nor google-generativeai is installed.")


# ==============================================================================
# 2. GEMINI API INVOCATION & ERROR CLASSIFICATION
# ==============================================================================
def is_transient_error(exception):
    """
    Checks if an exception indicates a 503 (UNAVAILABLE / high demand)
    or 429 (RATE_LIMIT / RESOURCE_EXHAUSTED) temporary condition.
    """
    err_str = str(exception).lower()
    transient_indicators = [
        "503", "unavailable", "high demand", "overloaded",
        "429", "resource_exhausted", "rate limit", "too many requests", "quota exceeded"
    ]
    return any(ind in err_str for ind in transient_indicators)


def call_gemini_api(client, model_name, temperature, prompt_text):
    """
    Calls Gemini API with the given prompt and requests structured JSON output.
    """
    if USE_MODERN_SDK:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        return response.text
    else:
        model = client.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        )
        response = model.generate_content(prompt_text)
        return response.text


def extract_and_parse_json(raw_text):
    """
    Strips code fences if present and parses raw string into a Python JSON structure.
    """
    if not raw_text:
        return None

    cleaned = raw_text.strip()

    # Remove markdown code fences like ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None


# ==============================================================================
# 3. VALIDATION, EXPONENTIAL BACKOFF & MODEL FALLBACK
# ==============================================================================
def validate_batch_response(parsed_data, expected_count):
    """
    Validates that the parsed response:
    1. Is a list.
    2. Has the exact expected number of items matching the batch.
    3. Contains all required fields with proper types:
       - normalized: string
       - sentiment: 'positive', 'negative', or 'neutral'
       - sarcasm: boolean
       - aspects: list
       - confidence: float between 0 and 1
    """
    if not isinstance(parsed_data, list):
        return False, f"Expected JSON list, got {type(parsed_data).__name__}"

    if len(parsed_data) != expected_count:
        return False, f"Expected {expected_count} items in batch response, received {len(parsed_data)}"

    valid_sentiments = {"positive", "negative", "neutral"}

    for idx, item in enumerate(parsed_data):
        if not isinstance(item, dict):
            return False, f"Item at index {idx} is not a dictionary"

        required_fields = ["normalized", "sentiment", "sarcasm", "aspects", "confidence"]
        for field in required_fields:
            if field not in item:
                return False, f"Item {idx} is missing required field '{field}'"

        sentiment = str(item.get("sentiment", "")).lower()
        if sentiment not in valid_sentiments:
            return False, f"Item {idx} has invalid sentiment '{sentiment}' (expected positive/negative/neutral)"

        if not isinstance(item.get("sarcasm"), bool):
            if str(item.get("sarcasm")).lower() in ["true", "false"]:
                item["sarcasm"] = str(item.get("sarcasm")).lower() == "true"
            else:
                return False, f"Item {idx} has non-boolean sarcasm: {item.get('sarcasm')}"

        if not isinstance(item.get("aspects"), list):
            return False, f"Item {idx} 'aspects' must be a list"

        try:
            conf = float(item.get("confidence", 0.0))
            item["confidence"] = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            return False, f"Item {idx} has invalid confidence value: {item.get('confidence')}"

    return True, "Valid"


def process_batch_with_retry(
    client,
    model_name,
    temperature,
    prompt_template,
    batch_items,
    retry_limit=1,
    fallback_models=None,
    raise_on_failure=False,
):
    """
    Processes a batch of items with:
    1. Exponential backoff (2s, 4s, 8s, 16s) for 503 and 429 transient errors.
    2. Model fallback: tries models in fallback_models if the primary model fails.
    3. Handles failure cleanly: raises AIServiceBusyError if raise_on_failure=True,
       or marks sentiment as 'ERROR' in batch CSV flow.
    """
    batch_payload = [
        {"id": item["id"], "text": item["original"]}
        for item in batch_items
    ]
    batch_json_str = json.dumps(batch_payload, ensure_ascii=False, indent=2)
    prompt_text = prompt_template.replace("{batch_json}", batch_json_str)

    # Build sequence of models to try
    models_to_try = [model_name]
    if fallback_models:
        for fb_model in fallback_models:
            if fb_model not in models_to_try:
                models_to_try.append(fb_model)

    # Exponential backoff schedule for 503/429 transient errors (4 retries)
    backoff_delays = [2, 4, 8, 16]
    last_error = ""

    for m_idx, current_model in enumerate(models_to_try):
        if m_idx > 0:
            print(f"  [Model Fallback] Switching to model '{current_model}' (candidate {m_idx + 1}/{len(models_to_try)})...")

        attempt = 0
        while attempt <= len(backoff_delays):
            try:
                raw_response = call_gemini_api(client, current_model, temperature, prompt_text)
                parsed_data = extract_and_parse_json(raw_response)

                if parsed_data is not None:
                    is_valid, validation_msg = validate_batch_response(parsed_data, len(batch_items))
                    if is_valid:
                        return parsed_data
                    else:
                        last_error = validation_msg
                        print(f"  [{current_model}] JSON validation failed: {validation_msg}")
                        # Retry once for JSON validation issue
                        if attempt == 0:
                            attempt += 1
                            continue
                else:
                    last_error = "Could not parse response as valid JSON"
                    print(f"  [{current_model}] JSON parsing failed.")

            except Exception as e:
                last_error = str(e)

                # Check for 503 UNAVAILABLE or 429 RATE_LIMIT
                if is_transient_error(e):
                    if attempt < len(backoff_delays):
                        wait_time = backoff_delays[attempt]
                        print(f"  [{current_model} | Attempt {attempt + 1}] Transient error (503/429). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                    else:
                        print(f"  [{current_model}] Reached max retries ({len(backoff_delays)}) for 503/429 errors.")
                        break  # Move to next fallback model
                else:
                    # Non-transient error (e.g., model 404, invalid param, etc.)
                    print(f"  [{current_model}] Non-transient error: {e}")
                    break  # Move to next fallback model

            attempt += 1

    # All models and retries exhausted
    if raise_on_failure:
        raise AIServiceBusyError("The AI service is busy, please try again in a minute.")

    # In batch CSV mode, mark failed rows as 'ERROR' instead of 'neutral'
    print(f"  [Warning] All models and retries failed: {last_error}. Marking rows as ERROR.")
    fallback_results = []
    for item in batch_items:
        fallback_results.append({
            "id": item["id"],
            "normalized": "",
            "sentiment": "ERROR",
            "sarcasm": False,
            "aspects": [],
            "confidence": 0.0,
            "notes": "The AI service is busy, please try again in a minute."
        })
    return fallback_results


# ==============================================================================
# 4. HIGH-LEVEL HELPERS (SINGLE TEXT & DATAFRAME BATCH PROCESSING)
# ==============================================================================
def process_single_text(text, client, config=None, prompt_template=None):
    """
    Processes a single Hinglish text string.
    Raises AIServiceBusyError if all models and backoff retries fail,
    preventing fake Neutral / 0% confidence cards in the UI.
    """
    if not text or not str(text).strip():
        raise ValueError("Input text cannot be empty.")

    if config is None:
        config = load_config()
    if prompt_template is None:
        prompt_template = load_prompt_template(config["prompt_path"])

    item = {"id": 1, "original": str(text).strip()}
    results = process_batch_with_retry(
        client=client,
        model_name=config["model_name"],
        temperature=config["temperature"],
        prompt_template=prompt_template,
        batch_items=[item],
        retry_limit=config.get("retry_limit", 1),
        fallback_models=config.get("fallback_models", []),
        raise_on_failure=True,
    )

    res = results[0]
    return {
        "original": item["original"],
        "normalized": res.get("normalized", ""),
        "sentiment": res.get("sentiment", "neutral"),
        "sarcasm": bool(res.get("sarcasm", False)),
        "aspects": res.get("aspects", []),
        "confidence": round(float(res.get("confidence", 0.0)), 2),
        "notes": res.get("notes", "")
    }


def process_dataframe(df, client, config=None, prompt_template=None, progress_callback=None):
    """
    Processes a pandas DataFrame containing Hinglish texts in batches.
    Failed batches are marked as 'ERROR' instead of 'neutral'.
    """
    if config is None:
        config = load_config()
    if prompt_template is None:
        prompt_template = load_prompt_template(config["prompt_path"])

    text_col = None
    for col in ["text", "tweet", "review", "comment", "content"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]

    items_to_process = []
    final_rows = []

    for idx, row in df.iterrows():
        raw_val = row[text_col]
        if pd.isna(raw_val) or str(raw_val).strip() == "":
            final_rows.append({
                "row_index": idx,
                "original": "" if pd.isna(raw_val) else str(raw_val),
                "normalized": "",
                "sentiment": "neutral",
                "sarcasm": False,
                "aspects": json.dumps([]),
                "confidence": 1.0,
                "notes": "Empty input row skipped"
            })
        else:
            items_to_process.append({
                "id": len(items_to_process) + 1,
                "row_index": idx,
                "original": str(raw_val).strip()
            })

    batch_size = config["batch_size"]
    total_batches = (len(items_to_process) + batch_size - 1) // batch_size if items_to_process else 0

    for b_idx in range(total_batches):
        batch = items_to_process[b_idx * batch_size : (b_idx + 1) * batch_size]

        if progress_callback:
            progress_callback(b_idx + 1, total_batches, (b_idx + 1) / total_batches)

        batch_results = process_batch_with_retry(
            client=client,
            model_name=config["model_name"],
            temperature=config["temperature"],
            prompt_template=prompt_template,
            batch_items=batch,
            retry_limit=config.get("retry_limit", 1),
            fallback_models=config.get("fallback_models", []),
            raise_on_failure=False,
        )

        for item, result in zip(batch, batch_results):
            aspects_data = result.get("aspects", [])
            aspects_str = json.dumps(aspects_data, ensure_ascii=False) if isinstance(aspects_data, list) else str(aspects_data)

            final_rows.append({
                "row_index": item["row_index"],
                "original": item["original"],
                "normalized": result.get("normalized", ""),
                "sentiment": result.get("sentiment", "ERROR"),
                "sarcasm": bool(result.get("sarcasm", False)),
                "aspects": aspects_str,
                "confidence": round(float(result.get("confidence", 0.0)), 2),
                "notes": result.get("notes", "")
            })

    final_rows.sort(key=lambda r: r["row_index"])
    output_columns = ["original", "normalized", "sentiment", "sarcasm", "aspects", "confidence", "notes"]
    return pd.DataFrame(final_rows)[output_columns]
