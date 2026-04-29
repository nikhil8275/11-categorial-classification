from flask import Flask, render_template, request, jsonify
from model_predictor import predict_website, predict_batch
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'Please provide a URL'}), 400

    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        result = predict_website(url)
        # result is already a dict with status, url, predictions, [note]
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'url': url,
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/predict-batch', methods=['POST'])
def batch_predict():
    data = request.get_json()
    urls = data.get('urls', [])

    if not urls:
        return jsonify({'error': 'Please provide at least one URL'}), 400

    normalized_urls = []
    for url in urls:
        url = url.strip()
        if url and not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        normalized_urls.append(url)

    try:
        results = predict_batch(normalized_urls)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

