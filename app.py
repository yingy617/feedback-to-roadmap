import os
import re
import json
import time
import logging
import threading
from html import escape as html_escape
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('feedback-to-roadmap')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp'

# Initialize Anthropic client
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")
client = Anthropic(api_key=api_key)

MODEL = os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-6')
# Max concurrent Claude calls for per-theme work. 1 = sequential.
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '8'))
# USD per million tokens, used for the cost estimate. Override if pricing changes.
PRICE_INPUT_PER_MTOK = float(os.getenv('PRICE_INPUT_PER_MTOK', '3.0'))
PRICE_OUTPUT_PER_MTOK = float(os.getenv('PRICE_OUTPUT_PER_MTOK', '15.0'))

VALID_SENTIMENTS = {'positive', 'negative', 'neutral'}
VALID_SEVERITIES = {'low', 'medium', 'high'}


# ---------------------------------------------------------------------------
# Metrics: tokens, latency, cost
# ---------------------------------------------------------------------------

class RunMetrics:
    """Collects per-call token usage and latency for one pipeline stage (thread-safe)."""

    def __init__(self, stage):
        self.stage = stage
        self.calls = []
        self._lock = threading.Lock()
        self._start = time.perf_counter()

    def record(self, label, latency_s, input_tokens, output_tokens):
        with self._lock:
            self.calls.append({
                'label': label,
                'latency_ms': round(latency_s * 1000),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
            })

    def summary(self):
        wall_ms = round((time.perf_counter() - self._start) * 1000)
        inp = sum(c['input_tokens'] for c in self.calls)
        out = sum(c['output_tokens'] for c in self.calls)
        cost = inp / 1e6 * PRICE_INPUT_PER_MTOK + out / 1e6 * PRICE_OUTPUT_PER_MTOK
        return {
            'stage': self.stage,
            'wall_ms': wall_ms,
            'llm_calls': len(self.calls),
            'sum_call_latency_ms': sum(c['latency_ms'] for c in self.calls),
            'input_tokens': inp,
            'output_tokens': out,
            'cost_usd': round(cost, 5),
            'calls': self.calls,
        }

    def log_summary(self):
        s = self.summary()
        log.info('[%s] wall=%dms calls=%d sum_call_latency=%dms tokens in=%d out=%d cost=$%.4f',
                 s['stage'], s['wall_ms'], s['llm_calls'], s['sum_call_latency_ms'],
                 s['input_tokens'], s['output_tokens'], s['cost_usd'])
        return s


def call_claude(prompt, max_tokens, metrics, label):
    """Single Claude call that records tokens and latency from the response's usage data."""
    t0 = time.perf_counter()
    message = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = time.perf_counter() - t0
    usage = message.usage
    metrics.record(label, latency, usage.input_tokens, usage.output_tokens)
    log.info('[%s] %s: %dms in=%d out=%d', metrics.stage, label, latency * 1000,
             usage.input_tokens, usage.output_tokens)
    return message.content[0].text


def run_parallel(fn, items, workers=None):
    """Map fn over items, concurrently when workers > 1. Preserves order."""
    workers = MAX_WORKERS if workers is None else workers
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))


def extract_json(text):
    start = text.find('{')
    end = text.rfind('}') + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    return json.loads(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_feedback(text):
    """Split feedback text into individual items, filtering empty lines."""
    items = [line.strip() for line in text.split('\n') if line.strip()]
    return items


def valid_indices(indices, all_items):
    """Keep only 1-based indices that point at a real feedback item."""
    out = []
    for idx in indices or []:
        try:
            i = int(idx)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= len(all_items) and i not in out:
            out.append(i)
    return out


_QUOTE_TRANSLATION = str.maketrans({
    '“': '"', '”': '"', '‘': "'", '’': "'",
    '–': '-', '—': '-', ' ': ' ',
})


def normalize_text(s):
    """Normalize for quote matching: smart quotes, case, whitespace, edge punctuation."""
    s = (s or '').translate(_QUOTE_TRANSLATION).lower()
    s = re.sub(r'\s+', ' ', s).strip()
    return s.strip(' "\'.,;:!?')


def quote_in_text(quote, text):
    """True if the quote appears verbatim (after normalization) in text.
    Quotes with an ellipsis must have every fragment present, in order."""
    fragments = [normalize_text(f) for f in re.split(r'\.\.\.|…', quote)]
    fragments = [f for f in fragments if f]
    if not fragments:
        return False
    target = normalize_text(text)
    pos = 0
    for frag in fragments:
        found = target.find(frag, pos)
        if found < 0:
            return False
        pos = found + len(frag)
    return True


def verify_quotes(quotes, theme_items, all_items):
    """Deterministic check: does each quote actually appear in the source feedback?"""
    theme_texts = [all_items[i - 1] for i in valid_indices(theme_items, all_items)]
    results = []
    for q in quotes or []:
        if any(quote_in_text(q, t) for t in theme_texts):
            results.append({'quote': q, 'verified': True, 'location': 'theme'})
        elif any(quote_in_text(q, t) for t in all_items):
            results.append({'quote': q, 'verified': True, 'location': 'other_theme'})
        else:
            results.append({'quote': q, 'verified': False, 'location': None})
    return results


# ---------------------------------------------------------------------------
# LLM steps
# ---------------------------------------------------------------------------

def check_recommendation_grounding(recommendation, supporting_quotes, theme_items, all_items,
                                   metrics, label='grounding'):
    """Verify a recommendation: deterministic quote check first, then LLM judgment."""
    quote_checks = verify_quotes(supporting_quotes, theme_items, all_items)
    verified = [c['quote'] for c in quote_checks if c['verified']]
    unverified = len(quote_checks) - len(verified)
    base = {
        'quote_checks': quote_checks,
        'quotes_verified': len(verified),
        'quotes_total': len(quote_checks),
    }

    if not recommendation or not supporting_quotes:
        return {**base, 'grounded': False, 'llm_checked': False,
                'status': 'Needs review — no supporting quotes',
                'reasoning': 'Recommendation has no supporting quotes'}

    if not verified:
        return {**base, 'grounded': False, 'llm_checked': False,
                'status': 'Needs review — quotes not found in source',
                'reasoning': f'None of the {len(quote_checks)} quotes appear verbatim in the feedback, '
                             'so the LLM check was skipped.'}

    # Only verified quotes go to the LLM: it judges support, code has already judged existence.
    quotes_text = '\n'.join([f'- "{quote}"' for quote in verified])
    theme_idx = valid_indices(theme_items, all_items)
    all_feedback_text = '\n'.join([f'{i+1}. {all_items[idx-1]}' for i, idx in enumerate(theme_idx)])

    prompt = f"""You are a quality assurance reviewer verifying that product recommendations are grounded in user feedback.

RECOMMENDATION:
{recommendation}

SUPPORTING QUOTES (already verified to appear verbatim in the feedback):
{quotes_text}

ALL RELATED FEEDBACK ITEMS (for context):
{all_feedback_text}

Task: Determine if the recommendation is clearly supported by the supporting quotes.

Criteria for "Grounded":
- The supporting quotes directly address the need/problem stated in the recommendation
- The recommendation logically follows from what users are saying

Criteria for "Needs review":
- Quotes don't clearly support the recommendation
- Recommendation overstates or invents claims not in the quotes
- Disconnect between quotes and recommendation

Return ONLY valid JSON:
{{
  "grounded": true or false,
  "reasoning": "Brief explanation of whether the recommendation is supported by its quotes"
}}"""

    try:
        response_text = call_claude(prompt, 400, metrics, label)
        result = extract_json(response_text)
    except json.JSONDecodeError:
        return {**base, 'grounded': False, 'llm_checked': False,
                'status': 'Needs review — unable to verify',
                'reasoning': 'Verification check failed'}

    grounded = bool(result.get('grounded'))
    if grounded:
        status = 'Grounded'
        if unverified:
            status += f' — {unverified} quote(s) removed, not found in source'
    else:
        status = 'Needs review — not clearly supported'
    return {**base, 'grounded': grounded, 'llm_checked': True,
            'status': status, 'reasoning': result.get('reasoning', '')}


def generate_recommendation(theme_name, theme_items, all_items, metrics, label='recommend'):
    """Use Claude to draft a prioritized recommendation for a theme with source quote grounding."""
    theme_idx = valid_indices(theme_items, all_items)
    if not theme_idx:
        return {'recommendation': '', 'supporting_quotes': []}

    items_text = '\n'.join([f'- {all_items[idx - 1]}' for idx in theme_idx])

    prompt = f"""You are a product manager synthesizing user feedback into prioritized recommendations.

Theme: {theme_name}
Related feedback items:
{items_text}

Generate:
1. A single, prioritized product recommendation (1-2 sentences, action-oriented)
2. A list of the most relevant quotes that support this recommendation

Return ONLY valid JSON with this exact structure:
{{
  "recommendation": "Specific, actionable recommendation here",
  "supporting_quotes": ["Exact quote from feedback item 1", "Exact quote from feedback item 2"]
}}

Be precise: supporting_quotes should be EXACT strings from the feedback items above, not paraphrased."""

    response_text = call_claude(prompt, 800, metrics, label)
    try:
        return extract_json(response_text)
    except json.JSONDecodeError:
        return {'recommendation': '', 'supporting_quotes': [], 'error': 'Failed to parse recommendation'}


def cluster_and_tag_feedback(items, metrics):
    """Use Claude to cluster feedback items into themes and tag sentiment/severity."""
    if not items:
        return {'themes': []}

    items_text = '\n'.join([f'{i+1}. {item}' for i, item in enumerate(items)])

    prompt = f"""Analyze this user feedback and:
1. Group items into thematic clusters (e.g., "Performance", "UI/UX", "Feature Requests")
2. For each theme, extract all related feedback items
3. Tag the overall theme with:
   - sentiment: "positive", "negative", or "neutral"
   - severity: "low", "medium", or "high" (based on impact/frequency)

Return ONLY valid JSON with this exact structure:
{{
  "themes": [
    {{
      "name": "Theme Name",
      "sentiment": "negative",
      "severity": "high",
      "items": [1, 2, 3]
    }}
  ]
}}

Items to analyze:
{items_text}"""

    response_text = call_claude(prompt, 2000, metrics, 'cluster')
    try:
        result = extract_json(response_text)
    except json.JSONDecodeError:
        return {'themes': [], 'error': 'Failed to parse Claude response'}

    # Sanitize model output so bad indices or tags can't break later steps.
    clean = []
    for t in result.get('themes', []):
        idx = valid_indices(t.get('items'), items)
        if not idx:
            continue
        clean.append({
            'name': str(t.get('name', 'Untitled')),
            'sentiment': t.get('sentiment') if t.get('sentiment') in VALID_SENTIMENTS else 'neutral',
            'severity': t.get('severity') if t.get('severity') in VALID_SEVERITIES else 'low',
            'items': idx,
        })
    return {'themes': clean}


def recommend_all(themes, items, metrics, workers=None):
    """Generate a recommendation per theme (in parallel). Mutates and returns themes."""
    def work(pair):
        i, theme = pair
        try:
            rec = generate_recommendation(theme['name'], theme['items'], items, metrics, f'recommend[{i}]')
        except Exception as e:  # one failed theme shouldn't sink the run
            log.exception('recommend failed for theme %r', theme.get('name'))
            rec = {'recommendation': '', 'supporting_quotes': [], 'error': str(e)}
        return rec

    recs = run_parallel(work, list(enumerate(themes)), workers)
    for theme, rec in zip(themes, recs):
        theme['recommendation'] = rec.get('recommendation', '')
        theme['supporting_quotes'] = rec.get('supporting_quotes', [])
        if rec.get('error'):
            theme['recommendation_error'] = rec['error']
    return themes


def ground_all(themes, items, metrics, workers=None):
    """Grounding check per theme (in parallel), then rank. Mutates and returns themes."""
    def work(pair):
        i, theme = pair
        try:
            return check_recommendation_grounding(
                theme.get('recommendation', ''), theme.get('supporting_quotes', []),
                theme.get('items', []), items, metrics, f'grounding[{i}]')
        except Exception as e:
            log.exception('grounding failed for theme %r', theme.get('name'))
            return {'grounded': False, 'llm_checked': False, 'quote_checks': [],
                    'status': 'Needs review — unable to verify', 'reasoning': str(e)}

    results = run_parallel(work, list(enumerate(themes)), workers)
    severity_scores = {'low': 1, 'medium': 2, 'high': 3}
    for theme, g in zip(themes, results):
        theme['grounding_status'] = g['status']
        theme['grounding_check'] = g['grounded']
        theme['grounding_reasoning'] = g.get('reasoning', '')
        theme['llm_checked'] = g.get('llm_checked', False)
        theme['quote_checks'] = g.get('quote_checks', [])
        if 'approval_status' not in theme:
            theme['approval_status'] = None
        # Rank by frequency x severity
        theme['rank_score'] = len(theme.get('items', [])) * severity_scores.get(theme.get('severity', 'low'), 1)

    themes.sort(key=lambda t: t.get('rank_score', 0), reverse=True)
    return themes


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/parse', methods=['POST'])
def parse():
    items = []

    # Pasted text and an uploaded file are combined, not one replacing the other.
    if 'feedback_text' in request.form and request.form['feedback_text'].strip():
        items.extend(parse_feedback(request.form['feedback_text']))

    if 'feedback_file' in request.files:
        file = request.files['feedback_file']
        if file and file.filename:
            try:
                content = file.read().decode('utf-8', errors='ignore')
                items.extend(parse_feedback(content))
            except Exception as e:
                return jsonify({'error': f'Failed to read file: {str(e)}'}), 400

    if not items:
        return jsonify({'error': 'No feedback items found. Please paste text or upload a file.'}), 400

    return jsonify({
        'items': items,
        'count': len(items)
    })


@app.route('/cluster', methods=['POST'])
def cluster():
    """Cluster and tag feedback items using Claude."""
    try:
        data = request.get_json()
        items = data.get('items', [])

        if not items:
            return jsonify({'error': 'No items to cluster'}), 400

        metrics = RunMetrics('cluster')
        result = cluster_and_tag_feedback(items, metrics)
        result['metrics'] = metrics.log_summary()

        if 'error' in result:
            return jsonify(result), 400

        result['original_items'] = items
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Clustering failed: {str(e)}'}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    """Generate recommendations for clustered themes."""
    try:
        data = request.get_json()
        themes = data.get('themes', [])
        items = data.get('items', [])

        if not themes or not items:
            return jsonify({'error': 'No themes or items provided'}), 400

        metrics = RunMetrics('recommend')
        recommend_all(themes, items, metrics)
        return jsonify({'themes': themes, 'items': items, 'metrics': metrics.log_summary()})

    except Exception as e:
        return jsonify({'error': f'Recommendation generation failed: {str(e)}'}), 500


@app.route('/check-grounding', methods=['POST'])
def check_grounding():
    """Verify that recommendations are grounded in their supporting quotes."""
    try:
        data = request.get_json()
        themes = data.get('themes', [])
        items = data.get('items', [])

        if not themes or not items:
            return jsonify({'error': 'No themes or items provided'}), 400

        metrics = RunMetrics('grounding')
        ground_all(themes, items, metrics)
        return jsonify({'themes': themes, 'items': items, 'metrics': metrics.log_summary()})

    except Exception as e:
        return jsonify({'error': f'Grounding check failed: {str(e)}'}), 500


@app.route('/export', methods=['POST'])
def export():
    """Export approved recommendations as an HTML document."""
    try:
        data = request.get_json()
        themes = data.get('themes', [])

        # Filter to only approved themes
        approved_themes = [t for t in themes if t.get('approval_status') == 'approved']

        if not approved_themes:
            return jsonify({'error': 'No approved recommendations to export'}), 400

        # Generate HTML document
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feedback to Roadmap - PRD</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f9fafb;
            color: #374151;
            line-height: 1.6;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            color: #111827;
            font-size: 32px;
        }}
        .header p {{
            margin: 8px 0 0 0;
            color: #6b7280;
            font-size: 16px;
        }}
        .recommendation {{
            background: white;
            padding: 24px;
            margin-bottom: 24px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        .rec-title {{
            font-size: 20px;
            font-weight: 700;
            color: #111827;
            margin: 0 0 12px 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .rec-meta {{
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .badge-sentiment-positive {{ background: #dcfce7; color: #166534; }}
        .badge-sentiment-negative {{ background: #fee2e2; color: #991b1b; }}
        .badge-sentiment-neutral {{ background: #f3f4f6; color: #4b5563; }}
        .badge-severity-low {{ background: #dbeafe; color: #1e40af; }}
        .badge-severity-medium {{ background: #fef3c7; color: #b45309; }}
        .badge-severity-high {{ background: #fecaca; color: #991b1b; }}
        .rec-text {{
            padding: 14px;
            background: #fef3c7;
            border-left: 3px solid #f59e0b;
            border-radius: 4px;
            margin-bottom: 16px;
            font-size: 15px;
            line-height: 1.6;
        }}
        .rec-label {{
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #9ca3af;
            margin-top: 16px;
            margin-bottom: 8px;
        }}
        .quote {{
            padding: 12px;
            background: #e0e7ff;
            border-left: 3px solid #6366f1;
            border-radius: 4px;
            font-size: 13px;
            color: #312e81;
            font-style: italic;
            margin-bottom: 8px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
            color: #9ca3af;
            font-size: 12px;
        }}
        @media print {{
            body {{ background: white; }}
            .recommendation {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Product Recommendations</h1>
        <p>Generated from user feedback analysis</p>
    </div>
"""

        # Add each approved recommendation
        for theme in approved_themes:
            sentiment = theme.get('sentiment')
            sentiment = sentiment if sentiment in VALID_SENTIMENTS else 'neutral'
            severity = theme.get('severity')
            severity = severity if severity in VALID_SEVERITIES else 'low'

            # Only export quotes that passed the deterministic source check (when it ran).
            if theme.get('quote_checks'):
                quotes = [c['quote'] for c in theme['quote_checks'] if c.get('verified')]
            else:
                quotes = theme.get('supporting_quotes') or []

            quotes_html = ''
            for quote in quotes:
                quotes_html += f'<div class="quote">"{html_escape(str(quote))}"</div>\n'

            grounding_indicator = ''
            if theme.get('grounding_check'):
                grounding_indicator = ' ✓'

            html += f"""    <div class="recommendation">
        <div class="rec-title">{html_escape(str(theme.get('name', '')))}{grounding_indicator}</div>
        <div class="rec-meta">
            <span class="badge badge-sentiment-{sentiment}">{sentiment}</span>
            <span class="badge badge-severity-{severity}">{severity}</span>
        </div>
        <div class="rec-text">{html_escape(str(theme.get('recommendation', '')))}</div>
        <div class="rec-label">Supporting Evidence</div>
        {quotes_html}
    </div>
"""

        html += """    <div class="footer">
        <p>Generated by Feedback to Roadmap</p>
    </div>
</body>
</html>"""

        return html, 200, {'Content-Type': 'text/html', 'Content-Disposition': 'attachment; filename="roadmap.html"'}

    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, port=5001)
