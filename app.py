"""
Hinglish Normalizer + Sentiment Analysis - Streamlit Web UI
------------------------------------------------------------
A clean, interactive Streamlit dashboard for normalizing and analyzing
Hinglish (Hindi-English code-mixed) texts using Google Gemini.
Supports both single-text interactive analysis and batch CSV processing
with resilient error handling (exponential backoff & fallback models).
"""

import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Import and reload the reusable processing engine from processor.py to prevent stale memory cache
import importlib
import processor
try:
    importlib.reload(processor)
except Exception:
    pass

from processor import (
    load_config,
    load_prompt_template,
    init_gemini_client,
    process_single_text,
    process_dataframe,
)

# Safe import with fallback to prevent caching conflicts in Streamlit
try:
    from processor import AIServiceBusyError
except ImportError:
    class AIServiceBusyError(Exception):
        """Fallback in case of stale module cache."""
        pass

# ==============================================================================
# 1. PAGE SETUP & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Hinglish Normalizer + Sentiment",
    page_icon="🇮🇳",
    layout="wide",
)

# Custom CSS for modern card designs, badges, and tag styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e9ecef;
        margin-bottom: 12px;
    }
    .badge-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #e2e3e5;
        color: #383d41;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .aspect-pill {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 15px;
        padding: 4px 12px;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
        font-size: 0.9rem;
    }
    .sarcasm-alert {
        background-color: #fff3cd;
        color: #856404;
        padding: 6px 12px;
        border-radius: 6px;
        border-left: 4px solid #ffeeba;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. INITIALIZATION (CONFIG, PROMPT, API KEY)
# ==============================================================================
load_dotenv()

def get_resources():
    """Loads current configuration and prompt template freshly from disk."""
    cfg = load_config("config.yaml")
    prompt = load_prompt_template(cfg["prompt_path"])
    return cfg, prompt

try:
    config, prompt_template = get_resources()
except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

# Sidebar: Controls & API Status
st.sidebar.title("⚙️ Settings")
st.sidebar.write(f"**Primary Model:** `{config['model_name']}`")
fallback_list = config.get("fallback_models", [])
if fallback_list:
    st.sidebar.write(f"**Fallback Models:** {', '.join([f'`{m}`' for m in fallback_list])}")
st.sidebar.write(f"**Temperature:** `{config['temperature']}`")
st.sidebar.write(f"**Batch Size:** `{config['batch_size']}`")

env_api_key = os.getenv("GEMINI_API_KEY", "")
user_api_key = st.sidebar.text_input(
    "Gemini API Key (optional override):",
    value=env_api_key,
    type="password",
    help="Leave blank to use the GEMINI_API_KEY from your .env file"
)
active_api_key = user_api_key.strip() if user_api_key.strip() else env_api_key

if active_api_key:
    st.sidebar.success("API Key detected")
else:
    st.sidebar.warning("No API Key detected. Please enter it above or in `.env`")

# Helper to instantiate Gemini client safely
def get_gemini_client():
    if not active_api_key:
        st.error("Missing Gemini API Key! Please enter it in the sidebar or in your `.env` file.")
        return None
    try:
        return init_gemini_client(active_api_key)
    except Exception as err:
        st.error(f"Failed to initialize Gemini Client: {err}")
        return None


# ==============================================================================
# 3. HEADER & DESCRIPTION
# ==============================================================================
st.title("Hinglish Normalizer + Sentiment Analysis")
st.caption(
    "Translate code-mixed Hinglish into clean standard English, detect sarcasm, "
    "and perform aspect-based sentiment analysis using Google Gemini."
)
st.write("---")

# ==============================================================================
# 4. TABS: SINGLE TEXT & UPLOAD CSV
# ==============================================================================
tab_single, tab_csv = st.tabs(["📝 Single Text", "📁 Upload CSV"])

# ------------------------------------------------------------------------------
# TAB 1: SINGLE TEXT ANALYSIS
# ------------------------------------------------------------------------------
with tab_single:
    st.subheader("Analyze an Individual Hinglish Sentence")

    # Sample Hinglish texts for quick testing
    samples = {
        "📱 Camera vs Battery": "Phone ka camera ekdum mast hai yaar, but battery thodi jaldi drain ho jati hai.",
        "🚕 Sarcastic Cab Delay": "Wah bhai 2 ghante late aakar bolte ho traffic tha, best cab service ever!",
        "🍛 Butter Chicken Praise": "Bhai restaurant ka ambience aur butter chicken dono ekdum bawa the, full paisa vasool!"
    }

    # Example Buttons
    st.write("**Try a sample text:**")
    col1, col2, col3 = st.columns(3)

    if "single_input_text" not in st.session_state:
        st.session_state["single_input_text"] = samples["📱 Camera vs Battery"]

    if col1.button("📱 Camera vs Battery"):
        st.session_state["single_input_text"] = samples["📱 Camera vs Battery"]
    if col2.button("🚕 Sarcastic Cab Delay"):
        st.session_state["single_input_text"] = samples["🚕 Sarcastic Cab Delay"]
    if col3.button("🍛 Butter Chicken Praise"):
        st.session_state["single_input_text"] = samples["🍛 Butter Chicken Praise"]

    # Input Text Area
    user_input = st.text_area(
        "Enter Hinglish tweet, review, or message:",
        value=st.session_state["single_input_text"],
        height=100,
        placeholder="Type your Hinglish text here..."
    )

    # Analyze Button
    if st.button("🔍 Analyze Text", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some Hinglish text before analyzing.")
        else:
            client = get_gemini_client()
            if client:
                with st.spinner("Analyzing text with Gemini (with backoff & model fallback)..."):
                    try:
                        result = process_single_text(
                            text=user_input,
                            client=client,
                            config=config,
                            prompt_template=prompt_template
                        )

                        # Display Result Card only if processing succeeded
                        st.write("---")
                        st.subheader("📊 Analysis Results")

                        res_col1, res_col2 = st.columns(2)

                        with res_col1:
                            st.markdown("**Original Hinglish Text:**")
                            st.info(result["original"])

                        with res_col2:
                            st.markdown("**Normalized English Translation:**")
                            st.success(result["normalized"])

                        # Metrics & Badges Row
                        metric_col1, metric_col2, metric_col3 = st.columns([1, 1, 2])

                        with metric_col1:
                            st.markdown("**Overall Sentiment:**")
                            sentiment = result["sentiment"].lower()
                            if sentiment == "positive":
                                st.markdown('<span class="badge-positive">🟢 Positive</span>', unsafe_allow_html=True)
                            elif sentiment == "negative":
                                st.markdown('<span class="badge-negative">🔴 Negative</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-neutral">⚪ Neutral</span>', unsafe_allow_html=True)

                        with metric_col2:
                            st.markdown("**Sarcasm Detection:**")
                            if result["sarcasm"]:
                                st.markdown('<span class="sarcasm-alert">🚨 Sarcastic / Mocking</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span>✅ Genuine (Not Sarcastic)</span>', unsafe_allow_html=True)

                        with metric_col3:
                            confidence_val = float(result.get("confidence", 0.0))
                            st.markdown(f"**Confidence:** `{int(confidence_val * 100)}%`")
                            st.progress(confidence_val)

                        # Aspects Extracted
                        st.markdown("**Extracted Aspects & Polarities:**")
                        aspects_list = result.get("aspects", [])
                        if aspects_list:
                            pills_html = ""
                            for asp in aspects_list:
                                a_name = asp.get("aspect", "general")
                                a_sent = asp.get("sentiment", "neutral").lower()
                                dot = "🟢" if a_sent == "positive" else ("🔴" if a_sent == "negative" else "⚪")
                                pills_html += f'<span class="aspect-pill"><b>{a_name}</b>: {dot} {a_sent.capitalize()}</span> '
                            st.markdown(pills_html, unsafe_allow_html=True)
                        else:
                            st.write("*(No specific product or service aspects evaluated)*")

                        # Ambiguity / Slang Notes
                        notes = result.get("notes", "")
                        if notes and notes.strip().lower() != "none":
                            st.markdown("**💡 Slang & Context Notes:**")
                            st.caption(f"ℹ️ {notes}")

                    except AIServiceBusyError:
                        # Clear message when all retries and fallback models fail (no fake Neutral card)
                        st.error("The AI service is busy, please try again in a minute.")
                    except Exception as err:
                        st.error(f"Analysis failed: {err}")


# ------------------------------------------------------------------------------
# TAB 2: UPLOAD CSV & BATCH PROCESSING
# ------------------------------------------------------------------------------
with tab_csv:
    st.subheader("Batch Process Hinglish CSV Dataset")
    st.write("Upload a CSV file containing a column with Hinglish texts (e.g. `text`, `tweet`, or `review`).")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.write(f"**File preview:** ({len(df_uploaded)} total rows)")
            st.dataframe(df_uploaded.head(5), use_container_width=True)

            # Detect text column
            possible_cols = [c for c in ["text", "tweet", "review", "comment", "content"] if c in df_uploaded.columns]
            default_col = possible_cols[0] if possible_cols else df_uploaded.columns[0]
            selected_col = st.selectbox("Select text column to process:", df_uploaded.columns, index=df_uploaded.columns.get_loc(default_col))

            if st.button("🚀 Process Batch CSV", type="primary"):
                client = get_gemini_client()
                if client:
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    def update_ui_progress(current_b, total_b, fraction):
                        progress_bar.progress(fraction)
                        status_text.text(f"Processing batch {current_b} of {total_b} ({int(fraction * 100)}%)...")

                    try:
                        df_to_run = df_uploaded.copy()
                        df_to_run["text"] = df_to_run[selected_col]

                        with st.spinner("Processing batches with Google Gemini..."):
                            df_results = process_dataframe(
                                df=df_to_run,
                                client=client,
                                config=config,
                                prompt_template=prompt_template,
                                progress_callback=update_ui_progress
                            )

                        status_text.success("Processing completed!")

                        # Summary Metrics
                        st.write("---")
                        st.subheader("📈 Results Summary")

                        total_count = len(df_results)
                        pos_count = (df_results["sentiment"] == "positive").sum()
                        neg_count = (df_results["sentiment"] == "negative").sum()
                        neu_count = (df_results["sentiment"] == "neutral").sum()
                        err_count = (df_results["sentiment"] == "ERROR").sum()
                        sarcasm_count = df_results["sarcasm"].sum()

                        m1, m2, m3, m4, m5, m6 = st.columns(6)
                        m1.metric("Total Rows", total_count)
                        m2.metric("Positive", pos_count)
                        m3.metric("Negative", neg_count)
                        m4.metric("Neutral", neu_count)
                        m5.metric("Errors", err_count)
                        m6.metric("Sarcastic", f"{sarcasm_count}")

                        if err_count > 0:
                            st.warning(f"⚠️ {err_count} row(s) could not be processed due to service busy errors and are marked as 'ERROR'. They have been excluded from the sentiment chart.")

                        # Sentiment Distribution Chart (EXCLUDES 'ERROR' rows)
                        valid_sentiments = df_results[df_results["sentiment"].isin(["positive", "negative", "neutral"])]
                        if not valid_sentiments.empty:
                            st.subheader("Sentiment Distribution")
                            sentiment_counts = valid_sentiments["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
                            st.bar_chart(sentiment_counts.set_index("sentiment"))

                        # Results Dataframe
                        st.subheader("Processed Results Table")
                        st.dataframe(df_results, use_container_width=True)

                        # CSV Download Button
                        csv_bytes = df_results.to_csv(index=False, encoding="utf-8").encode("utf-8")
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv_bytes,
                            file_name="hinglish_sentiment_results.csv",
                            mime="text/csv"
                        )

                    except Exception as err:
                        st.error(f"Error processing CSV: {err}")

        except Exception as e:
            st.error(f"Could not read uploaded CSV file: {e}")
