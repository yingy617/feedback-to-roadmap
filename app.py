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

if __name__ == '__main__':
    app.run(debug=False, port=5001)
