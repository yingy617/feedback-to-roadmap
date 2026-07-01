import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp'

# Initialize Anthropic client
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")
client = Anthropic(api_key=api_key)

def parse_feedback(text):
    """Split feedback text into individual items, filtering empty lines."""
    items = [line.strip() for line in text.split('\n') if line.strip()]
    return items

def check_recommendation_grounding(recommendation, supporting_quotes, theme_items, all_items):
    """Verify that a recommendation is supported by its source quotes."""
    if not recommendation or not supporting_quotes:
        return {
            'grounded': False,
            'status': 'Needs review — no supporting quotes',
            'reasoning': 'Recommendation has no supporting quotes'
        }

    # Format context for Claude
    quotes_text = '\n'.join([f'- "{quote}"' for quote in supporting_quotes])
    all_feedback_text = '\n'.join([f'{i+1}. {all_items[idx-1]}' for i, idx in enumerate(theme_items)])

    prompt = f"""You are a quality assurance reviewer verifying that product recommendations are grounded in user feedback.

RECOMMENDATION:
{recommendation}

SUPPORTING QUOTES (claimed to support the recommendation):
{quotes_text}

ALL RELATED FEEDBACK ITEMS (for context):
{all_feedback_text}

Task: Determine if the recommendation is clearly supported by the supporting quotes.

Criteria for "Grounded":
- The supporting quotes directly address the need/problem stated in the recommendation
- The recommendation logically follows from what users are saying
- The quotes are accurate/exact from the feedback

Criteria for "Needs review":
- Quotes don't clearly support the recommendation
- Recommendation overstates or invents claims not in the quotes
- Disconnect between quotes and recommendation

Return ONLY valid JSON:
{{
  "grounded": true or false,
  "reasoning": "Brief explanation of whether the recommendation is supported by its quotes"
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = message.content[0].text

    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            result = json.loads(json_str)
        else:
            result = json.loads(response_text)
    except json.JSONDecodeError:
        return {
            'grounded': False,
            'status': 'Needs review — unable to verify',
            'reasoning': 'Verification check failed'
        }

    status = 'Grounded' if result.get('grounded') else 'Needs review — not clearly supported'
    return {
        'grounded': result.get('grounded', False),
        'status': status,
        'reasoning': result.get('reasoning', '')
    }

def generate_recommendation(theme_name, theme_items, all_items):
    """Use Claude to draft a prioritized recommendation for a theme with source quote grounding."""
    if not theme_items:
        return {'recommendation': '', 'supporting_quotes': []}

    # Format items for Claude
    items_text = '\n'.join([f'- {all_items[idx - 1]}' for idx in theme_items])

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

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = message.content[0].text

    # Extract JSON from response
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            result = json.loads(json_str)
        else:
            result = json.loads(response_text)
    except json.JSONDecodeError:
        return {'recommendation': '', 'supporting_quotes': [], 'error': 'Failed to parse recommendation'}

    return result

def cluster_and_tag_feedback(items):
    """Use Claude to cluster feedback items into themes and tag sentiment/severity."""
    if not items:
        return {'themes': []}

    # Format items for Claude
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

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = message.content[0].text

    # Extract JSON from response
    try:
        # Try to find JSON in the response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            result = json.loads(json_str)
        else:
            result = json.loads(response_text)
    except json.JSONDecodeError:
        return {'themes': [], 'error': 'Failed to parse Claude response'}

    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/parse', methods=['POST'])
def parse():
    items = []

    # Handle text input
    if 'feedback_text' in request.form and request.form['feedback_text'].strip():
        text = request.form['feedback_text']
        items = parse_feedback(text)

    # Handle file upload
    if 'feedback_file' in request.files:
        file = request.files['feedback_file']
        if file and file.filename:
            try:
                content = file.read().decode('utf-8', errors='ignore')
                items = parse_feedback(content)
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

        result = cluster_and_tag_feedback(items)

        if 'error' in result:
            return jsonify(result), 400

        # Enhance result with original items for reference
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

        # Generate recommendation for each theme
        for theme in themes:
            rec = generate_recommendation(theme['name'], theme['items'], items)
            theme['recommendation'] = rec.get('recommendation', '')
            theme['supporting_quotes'] = rec.get('supporting_quotes', [])

        return jsonify({'themes': themes, 'items': items})

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

        # Check grounding for each theme
        for theme in themes:
            grounding = check_recommendation_grounding(
                theme.get('recommendation', ''),
                theme.get('supporting_quotes', []),
                theme.get('items', []),
                items
            )
            theme['grounding_status'] = grounding['status']
            theme['grounding_check'] = grounding['grounded']
            theme['grounding_reasoning'] = grounding.get('reasoning', '')
            # Initialize approval status
            if 'approval_status' not in theme:
                theme['approval_status'] = None

        # Rank themes by frequency × severity
        severity_scores = {'low': 1, 'medium': 2, 'high': 3}
        for theme in themes:
            frequency = len(theme.get('items', []))
            severity = severity_scores.get(theme.get('severity', 'low'), 1)
            theme['rank_score'] = frequency * severity

        # Sort by rank score descending
        themes.sort(key=lambda t: t.get('rank_score', 0), reverse=True)

        return jsonify({'themes': themes, 'items': items})

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
            sentiment = theme.get('sentiment', 'neutral')
            severity = theme.get('severity', 'low')

            quotes_html = ''
            if theme.get('supporting_quotes'):
                for quote in theme['supporting_quotes']:
                    quotes_html += f'<div class="quote">"{quote}"</div>\n'

            grounding_indicator = ''
            if theme.get('grounding_check'):
                grounding_indicator = ' ✓'

            html += f"""    <div class="recommendation">
        <div class="rec-title">{theme['name']}{grounding_indicator}</div>
        <div class="rec-meta">
            <span class="badge badge-sentiment-{sentiment}">{sentiment}</span>
            <span class="badge badge-severity-{severity}">{severity}</span>
        </div>
        <div class="rec-text">{theme.get('recommendation', '')}</div>
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
