# src/app/app.py
from flask import Flask, request, jsonify, render_template_string
from ..infer.infer import DiacriticPredictor

app = Flask(__name__)

# Initialize Predictor ONCE at startup
print("Loading model...")
predictor = DiacriticPredictor()
print("Model loaded!")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Arabic Diacritizer</title></head>
<body>
    <div style="max-width: 600px; margin: 50px auto; font-family: sans-serif;">
        <h2>Arabic Text Diacritization</h2>
        <textarea id="input_text" rows="4" style="width: 100%;" placeholder="أدخل النص العربي هنا..."></textarea>
        <br><br>
        <button onclick="diacritize()">Diacritize</button>
        <h3>Result:</h3>
        <p id="output_text" style="font-size: 1.2em; direction: rtl; border: 1px solid #ddd; padding: 10px; min-height: 50px;"></p>
    </div>

    <script>
        async function diacritize() {
            const text = document.getElementById('input_text').value;
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await response.json();
            document.getElementById('output_text').textContent = data.diacritized;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'diacritized': ''})
    
    result = predictor.predict(text)
    return jsonify({'diacritized': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)