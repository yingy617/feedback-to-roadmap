# Feedback to Roadmap

A web app that turns raw user feedback into prioritized product recommendations, grounded in real user quotes.

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone or navigate to the project directory:
```bash
cd /path/to/feedback-to-roadmap
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

```bash
python app.py
```

The app will start on `http://localhost:5001`

### Development

Flask is running in debug mode, so changes to `app.py` and `templates/` will auto-reload.

## How to Use

1. **Paste feedback** in the text box (one item per line), or
2. **Upload a file** (.txt or .csv) containing feedback items (one per line)
3. Click **Parse Feedback**
4. Review the parsed items

## Project Structure

```
feedback-to-roadmap/
├── app.py                 # Flask app
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # This file
└── templates/
    └── index.html        # Web UI
```

## API Endpoints

### POST /parse
Parses feedback and returns individual items.

**Request:**
- Form data with one of:
  - `feedback_text`: Raw text (one item per line)
  - `feedback_file`: File upload (.txt or .csv)

**Response:**
```json
{
  "items": ["Item 1", "Item 2", ...],
  "count": 2
}
```

Pasted text and an uploaded file are combined.

### POST /cluster, /recommend, /check-grounding, /export
Run the pipeline stages. The first three return a `metrics` object with `wall_ms`, `llm_calls`, `input_tokens`, `output_tokens`, `cost_usd` and per-call detail. `/check-grounding` also returns `quote_checks` per theme (`quote`, `verified`, `location`).

## Quality, Speed & Cost

- **Deterministic quote check.** Before the LLM grounding pass, code confirms every supporting quote actually appears in the source feedback (case, whitespace and smart-quote insensitive; `...` fragments must appear in order). Quotes that aren't found are flagged in the UI and dropped from the export, and only verified quotes are sent to the LLM. If none are verified, the LLM call is skipped. The model isn't trusted to grade itself on something code can verify.
- **Parallel per-theme calls.** Recommendations and grounding checks run concurrently in a thread pool (`MAX_WORKERS`, default 8; set to 1 for sequential).
- **Token and latency logging.** Every Claude call logs latency and `usage.input_tokens` / `usage.output_tokens`. Each endpoint returns a `metrics` object (wall time, calls, tokens, estimated cost), and the UI shows the running totals.

### Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Model used for all calls |
| `MAX_WORKERS` | `8` | Concurrent per-theme calls (1 = sequential) |
| `PRICE_INPUT_PER_MTOK` | `3.0` | USD per 1M input tokens, for cost estimates |
| `PRICE_OUTPUT_PER_MTOK` | `15.0` | USD per 1M output tokens, for cost estimates |

### Benchmark

```bash
python benchmark.py                      # uses sample_feedback.txt, 8 workers
python benchmark.py my_feedback.txt 4    # custom file / worker count
```

Runs clustering once, then recommend + grounding sequentially and in parallel, and prints latency, tokens, cost and how many quotes failed the source check. Makes real API calls (a few cents).

## Deploy to Production (Render)

### Prerequisites
- GitHub account (to host the code)
- Render account (free tier available at https://render.com)
- Claude API key from https://console.anthropic.com

### Steps

1. **Push to GitHub**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/feedback-to-roadmap.git
   git branch -M main
   git push -u origin main
   ```

2. **Create a Render Web Service**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - Configure:
     - **Name**: `feedback-to-roadmap` (or your choice)
     - **Environment**: Python 3
     - **Build command**: `pip install -r requirements.txt`
     - **Start command**: `gunicorn app:app`
     - **Region**: Choose closest to you

3. **Set Environment Variables**
   - In Render dashboard, go to your Web Service
   - Click "Environment" tab
   - Add new variable:
     - **Key**: `ANTHROPIC_API_KEY`
     - **Value**: Paste your Claude API key (never in code, only here)
   - Click "Save"

4. **Deploy**
   - Render auto-deploys when you push to GitHub
   - Your app will be live at: `https://feedback-to-roadmap.onrender.com`

### Important Security Notes
- **Never** commit `.env` file to GitHub
- **Always** set `ANTHROPIC_API_KEY` as an environment variable on your hosting platform
- `.env` is in `.gitignore` to prevent accidental commits
- Production uses `gunicorn`, local dev uses Flask dev server

## Milestones

- [x] Milestone 1: Input + item display
- [x] Milestone 2: LLM clustering + tagging
- [x] Milestone 3: Recommendations + source quotes
- [x] Milestone 4: Grounding verification
- [x] Milestone 5: Review UI (approval buttons, ranking) + deployment
