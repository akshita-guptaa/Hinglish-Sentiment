# Hinglish Normalizer + Sentiment Analysis

An end-to-end NLP pipeline and interactive **Streamlit Web Application** built with Python and the **Google Gemini API** that ingests Hinglish (Hindi-English code-mixed) social media posts, tweets, and product reviews, normalizes them into clean standard English, and extracts deep sentiment insights:
- **Normalization / Translation**: Translates colloquial Hinglish into fluent standard English.
- **Sentiment Classification**: Categorizes overall sentiment into `positive`, `negative`, or `neutral`.
- **Sarcasm Detection**: Flags whether sarcasm or mockery is present (`true`/`false`).
- **Aspect-Based Sentiment (ABSA)**: Extracts granular aspects (e.g., `camera`, `battery`, `delivery`, `price`) and their individual sentiment.
- **Confidence Scoring**: Assigns a confidence score between `0.0` and `1.0`.
- **Colloquial Notes**: Explains ambiguous slang or colloquial terms (e.g., *bawa*, *jugaad*, *paisa vasool*).

---

## 📁 Project Structure

```text
Hinglish-Sentiment/
│
├── config.yaml            # Pipeline configuration (model, batch size, temperature, paths)
├── processor.py           # Core reusable engine (API calls, validation, single/batch processing)
├── main.py                # Command-line entry point (reads input.csv, batches, exports results.csv)
├── app.py                 # Streamlit Web UI (Single text analysis & CSV upload dashboard)
├── requirements.txt       # Python dependencies (google-genai, streamlit, pandas, pyyaml, etc.)
├── .env.example           # Template for environment variables
├── .gitignore             # Ignores .env, __pycache__, output/, etc.
├── README.md              # Documentation and presentation guide
│
├── prompts/
│   └── normalize.txt      # Prompt template with instructions and 5 few-shot examples
│
├── data/
│   └── input.csv          # Sample input CSV containing raw Hinglish texts
│
└── output/
    └── results.csv        # Output CSV generated after processing
```

---

## 🚀 Setup & Installation

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd c:\Users\gupta\Desktop\projects\Hinglish-Sentiment
```

### Step 2: Create and Activate a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:

- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🔑 Setting Up Your Gemini API Key

1. Get an API key from [Google AI Studio](https://aistudio.google.com/).
2. Create a `.env` file in the root directory by copying `.env.example`:
   ```powershell
   Copy-Item .env.example .env
   ```
3. Open `.env` and set your key:
   ```env
   GEMINI_API_KEY="AIzaSy..."
   ```

> [!IMPORTANT]
> Never commit your `.env` file to Git. The `.gitignore` file is configured to keep your API key secure.

---

## 🌐 Running the Streamlit Web UI

You can launch the interactive web dashboard with:
```powershell
streamlit run app.py
```

The web app features two dedicated modes:
1. **📝 Single Text Analysis**:
   - Type any custom Hinglish text or click one of the 3 built-in sample buttons (*Camera vs Battery*, *Sarcastic Cab Delay*, *Butter Chicken Praise*).
   - Click **Analyze Text** to view:
     - Side-by-side original and normalized English translation.
     - Color-coded sentiment badge (🟢 Positive, 🔴 Negative, ⚪ Neutral).
     - Sarcasm alert indicator.
     - Confidence progress bar.
     - Aspect tags with individual sentiment indicators.
     - Nuance and slang notes.

2. **📁 Upload CSV & Batch Processing**:
   - Upload any CSV with a `text`, `tweet`, or `review` column.
   - Interactive batch progress bar showing percentage completion.
   - Summary metric cards (Total, Positive, Negative, Neutral, Sarcastic).
   - Sentiment distribution bar chart.
   - Download button to export the enriched CSV directly from your browser.

---

## ▶️ Running via Command-Line (`main.py`)

If you prefer batch processing via the CLI:
```powershell
python main.py
```
This reads [`data/input.csv`](file:///c:/Users/gupta/Desktop/projects/Hinglish-Sentiment/data/input.csv) in batches of 10, invokes Gemini, validates the responses, and saves the output to [`output/results.csv`](file:///c:/Users/gupta/Desktop/projects/Hinglish-Sentiment/output/results.csv).

---

## ⚙️ Configuration (`config.yaml`)

You can customize parameters in `config.yaml` without changing code:

```yaml
model_name: "gemini-3.8-flash"  # Primary model
fallback_models:               # Backup models if primary encounters errors
  - "gemini-2.5-flash"
  - "gemini-1.5-flash"
  - "gemini-1.5-pro"
temperature: 0.2               # Lower temperature ensures deterministic JSON output
batch_size: 10                 # Number of rows processed per Gemini API call
retry_limit: 1                 # Retries once if JSON validation fails
input_path: "data/input.csv"
output_path: "output/results.csv"
prompt_path: "prompts/normalize.txt"
```

---

## 📊 Output Schema

| Column | Description | Example |
| :--- | :--- | :--- |
| `original` | Original raw Hinglish text | `"Wah bhai 2 ghante late aakar..."` |
| `normalized` | Clean, fluent English translation | `"Wow brother, arriving 2 hours late and claiming there was traffic, best cab service ever!"` |
| `sentiment` | Overall sentiment (`positive`, `negative`, `neutral`, or `ERROR`) | `negative` |
| `sarcasm` | Boolean flag for sarcasm/irony | `True` |
| `aspects` | JSON list of extracted aspects and their polarity | `[{"aspect": "punctuality", "sentiment": "negative"}, {"aspect": "cab service", "sentiment": "negative"}]` |
| `confidence` | Model confidence score (0.0 to 1.0) | `0.98` |
| `notes` | Nuance or slang explanation | `"Sarcastic compliment used to express frustration with delay."` |

---

## 🎓 Student Explanation Guide (How It Works)

1. **Modular Architecture (`processor.py`)**:
   - Business logic is decoupled from presentation. Both `main.py` (CLI) and `app.py` (Streamlit UI) import from `processor.py`, eliminating code duplication and ensuring maintainability.

2. **Why Batching (10 rows per call)?**
   - Eliminates network overhead. Processing 100 rows in batches of 10 requires only 10 API requests instead of 100, saving latency and avoiding rate limits.

3. **Exponential Backoff for 503 / 429 Errors:**
   - When Gemini experiences temporary demand spikes (`503 UNAVAILABLE`) or rate limits (`429 RESOURCE_EXHAUSTED`), the pipeline pauses and retries up to 4 times with exponential backoff delays (`2s`, `4s`, `8s`, `16s`).

4. **Multi-Model Failover (`fallback_models`):**
   - If the primary model (`gemini-3.8-flash`) remains unavailable after all backoff retries, the pipeline automatically switches to backup models listed in `config.yaml` (e.g. `gemini-1.5-flash`, `gemini-1.5-pro`).

5. **Clean Error Handling & No Fake Data:**
   - In single-text UI mode: If all retries and fallback models fail, the app does **not** generate a fake neutral card; instead, it displays: *"The AI service is busy, please try again in a minute."*
   - In batch CSV mode: Failed rows are explicitly tagged as `ERROR` instead of `neutral`, and are cleanly excluded from the sentiment distribution chart.
