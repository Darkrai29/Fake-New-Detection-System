"""
Fake News Detection Web Application - Flask Server
Exposes model prediction, linguistic analysis, benchmark metrics,
and interactive web interface.
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from train_models import clean_text
from linguistic_analyzer import analyze_linguistics, extract_top_salient_tokens

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
METRICS_PATH = os.path.join(MODELS_DIR, 'model_metrics.json')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')

# Global variables for loaded artifacts
vectorizer = None
models = {}
metrics_data = {}
best_model_key = "SVM"

def load_resources():
    global vectorizer, models, metrics_data, best_model_key
    
    print("Loading metrics metadata...")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)
            best_model_key = metrics_data.get('best_model', 'SVM')
    else:
        print(f"Warning: {METRICS_PATH} not found.")

    print("Loading TF-IDF vectorizer...")
    if os.path.exists(VECTORIZER_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        print(f"Warning: {VECTORIZER_PATH} not found.")

    model_files = {
        'SVM': os.path.join(MODELS_DIR, 'svm.joblib'),
        'Random Forest': os.path.join(MODELS_DIR, 'random_forest.joblib'),
        'XGBoost': os.path.join(MODELS_DIR, 'xgboost.joblib')
    }

    for key, path in model_files.items():
        if os.path.exists(path):
            print(f"Loading {key} model from {path}...")
            models[key] = joblib.load(path)
        else:
            print(f"Warning: Model file for {key} not found at {path}")

# Load on startup
load_resources()

@app.route('/')
def index():
    return render_template('index.html', metrics=metrics_data, best_model=best_model_key)

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Returns the trained benchmark metrics for SVM, Random Forest, and XGBoost."""
    return jsonify({
        "status": "success",
        "data": metrics_data
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accepts an article (title, text, optional model_choice)
    Predicts FAKE NEWS or GENUINE NEWS, extracts linguistic features,
    and returns comprehensive explainability signals.
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form
        title = data.get('title', '').strip()
        text = data.get('text', '').strip()
        requested_model = data.get('model', best_model_key)

        if not text and not title:
            return jsonify({
                "status": "error",
                "message": "Please provide an article headline or body text."
            }), 400

        full_raw_text = f"{title} {text}".strip()
        cleaned = clean_text(full_raw_text)

        if not cleaned:
            return jsonify({
                "status": "error",
                "message": "The provided text contains only unsupported symbols or whitespace."
            }), 400

        # Feature extraction
        tfidf_vec = vectorizer.transform([cleaned])

        # Model resolution (fallback to best_model if requested doesn't exist)
        active_model_key = requested_model if requested_model in models else best_model_key
        clf = models[active_model_key]

        # Prediction: 1 -> FAKE, 0 -> GENUINE
        pred = int(clf.predict(tfidf_vec)[0])
        label = "FAKE NEWS" if pred == 1 else "GENUINE NEWS"

        # Probability estimate
        prob_fake = 0.5
        prob_genuine = 0.5
        confidence = 0.5

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(tfidf_vec)[0]
            prob_genuine = float(probs[0])
            prob_fake = float(probs[1])
            confidence = prob_fake if pred == 1 else prob_genuine
        elif hasattr(clf, "decision_function"):
            decision = float(clf.decision_function(tfidf_vec)[0])
            # Sigmoid approximation
            prob_fake = 1.0 / (1.0 + np.exp(-decision))
            prob_genuine = 1.0 - prob_fake
            confidence = prob_fake if pred == 1 else prob_genuine

        # Multi-model consensus comparison
        all_model_predictions = {}
        for m_key, m_clf in models.items():
            m_pred = int(m_clf.predict(tfidf_vec)[0])
            m_label = "FAKE NEWS" if m_pred == 1 else "GENUINE NEWS"
            m_conf = 0.5
            if hasattr(m_clf, "predict_proba"):
                m_probs = m_clf.predict_proba(tfidf_vec)[0]
                m_conf = float(m_probs[1]) if m_pred == 1 else float(m_probs[0])
            all_model_predictions[m_key] = {
                "label": m_label,
                "confidence": round(m_conf * 100, 2),
                "is_best": (m_key == best_model_key)
            }

        # Linguistic & Stylistic Analysis
        linguistics = analyze_linguistics(raw_text=text, title=title)

        # Salient predictive keywords from SVM
        svm_model = models.get('SVM')
        salient_tokens = extract_top_salient_tokens(full_raw_text, vectorizer, svm_model, top_k=8)

        return jsonify({
            "status": "success",
            "prediction": {
                "label": label,
                "confidence_percent": round(confidence * 100, 2),
                "prob_fake_percent": round(prob_fake * 100, 2),
                "prob_genuine_percent": round(prob_genuine * 100, 2),
                "model_used": active_model_key,
                "is_best_model": (active_model_key == best_model_key)
            },
            "multi_model_consensus": all_model_predictions,
            "linguistic_analysis": linguistics,
            "salient_tokens": salient_tokens
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/samples', methods=['GET'])
def get_samples():
    """Returns curated benchmark samples for immediate interactive testing."""
    samples = metrics_data.get('sample_articles', [])
    return jsonify({
        "status": "success",
        "samples": samples
    })

if __name__ == '__main__':
    print("Starting Fake News Detection Web Application on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
