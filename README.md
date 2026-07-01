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

## Milestones

- [x] Milestone 1: Input + item display (current)
- [ ] Milestone 2: LLM clustering + tagging
- [ ] Milestone 3: Recommendations + source quotes
- [ ] Milestone 4: Grounding check
- [ ] Milestone 5: Review UI + deploy
