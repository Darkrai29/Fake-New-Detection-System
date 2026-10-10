# 📰 Veritas AI: Fake News Detection & Linguistic Analysis System

A production-grade machine learning web application that predicts whether a news article is **FAKE NEWS** or **GENUINE NEWS** while conducting in-depth linguistic and textual characteristics analysis.

---

## 🚀 Key Features

1. **Complete Machine Learning Pipeline**:
   - **Text Preprocessing**: Normalization, HTML/URL stripping, regex cleanup, and tokenization.
   - **TF-IDF Feature Extraction**: 10,000 unigram and bigram features with sublinear term-frequency scaling.
   - **Three Competitive Classifiers Trained & Evaluated**:
     - **Support Vector Machine (SVM)** (LinearSVC with Platt probability calibration)
     - **Random Forest Classifier** (Ensemble of 150 decision trees)
     - **XGBoost Classifier** (Gradient boosted decision trees)
2. **Strict Ground-Truth Benchmark Evaluation**:
   - Zero invented metrics. All performance figures are calculated directly on **1,267 unseen test articles** from a stratified 80/20 split of the 6,335 benchmark corpus.
3. **Automated Best-Model Selection**:
   - Identifies the highest-performing model (**Linear SVM** with **95.13% F1-score**) and designates it for live prediction inference.
4. **Linguistic & Textual Characteristics Analysis**:
   - **Sensationalism & Clickbait Index**: Identifies sensationalist trigger words (e.g., *bombshell*, *conspiracy*, *hoax*).
   - **Journalistic Attribution Score**: Evaluates source attribution cues (e.g., *Reuters*, *spokesperson*, *confirmed*, *according to*).
   - **Typographical Anomalies**: Quantifies ALL-CAPS ratios, exclamation marks, and repeated punctuation.
   - **Lexical Diversity**: Measures Type-Token Ratio (TTR).
   - **Subjectivity / Pronoun Ratio**: Compares 1st/2nd person subjectivity against 3rd person journalistic neutrality.
   - **Explainable TF-IDF Keyword Signals**: Highlights the top mathematical n-grams swaying the classification.
5. **Interactive Web Application**:
   - Modern dark-mode interface with Chart.js metric comparison charts and interactive 2x2 confusion matrix explorer.
   - 1-click test buttons for pre-loaded genuine and fake news articles.
   - Multi-model consensus comparison showing individual model predictions side-by-side.

---

## 📊 Benchmark Model Performance (Calculated from Actual Dataset)

Dataset: `6,335 articles` (3,171 Genuine, 3,164 Fake).  
Split: `5,068 train` (80%), `1,267 test` (20%), stratified by class balance.

| Model | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TN, FP, FN, TP) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 👑 **Support Vector Machine (SVM)** | **95.11%** | **94.68%** | **95.58%** | **95.13%** | **TN=600, FP=34, FN=28, TP=605** | **0.87s** |
| ⚡ **XGBoost Classifier** | **93.05%** | **92.12%** | **94.15%** | **93.12%** | **TN=583, FP=51, FN=37, TP=596** | **159.89s** |
| 🌲 **Random Forest Classifier** | **90.77%** | **88.97%** | **93.05%** | **90.97%** | **TN=561, FP=73, FN=44, TP=589** | **6.13s** |

### 🏆 Best Model Selection Rationale
**Support Vector Machine (Linear SVM)** achieved the top performance across all evaluation dimensions:
- **Highest Accuracy**: 95.11%
- **Highest F1-Score**: 95.13%
- **Lowest Misclassification Count**: Only 62 errors out of 1,267 test articles (34 False Positives, 28 False Negatives).
- **Fastest Inference & Training**: Efficient convex optimization in high-dimensional TF-IDF space.

---

## 📁 Project Architecture

```
Fake News detection system/
├── app.py                      # Flask web server & REST API
├── train_models.py             # Preprocessing, TF-IDF, training & evaluation
├── linguistic_analyzer.py      # Stylistic, syntactic & sentiment explainability
├── test_app.py                 # Automated unit & integration test suite
├── news_dataset.csv            # Benchmark dataset (6,335 articles)
├── requirements.txt            # Python dependencies
├── run.bat                     # 1-click Windows runner
├── models/
│   ├── svm.joblib              # Calibrated Linear SVM model
│   ├── random_forest.joblib    # Random Forest classifier
│   ├── xgboost.joblib          # XGBoost classifier
│   ├── tfidf_vectorizer.joblib # Trained TF-IDF feature extractor (10k features)
│   └── model_metrics.json      # Ground-truth evaluation metrics & confusion matrices
├── templates/
│   └── index.html              # Dashboard & interactive classifier UI
└── static/
    ├── css/
    │   └── style.css           # Modern dark-mode styles
    └── js/
        └── app.js              # Chart.js visualizations & async prediction logic
```

---

## 🛠️ Installation & Setup

### 1. Requirements
- Python 3.10+ (Tested on Python 3.13)
- `pip install -r requirements.txt`

### 2. Retraining Models (Optional)
The models and metrics are already trained, evaluated, and saved in `models/`. To retrain from scratch:
```bash
python train_models.py
```

### 3. Running the Test Suite
```bash
python test_app.py
```

### 4. Running the Web Application
Double-click `run.bat` or run:
```bash
python app.py
```
Then open your web browser at:
```
http://127.0.0.1:5000
```

---

## 🔌 API Endpoints

- **`POST /api/predict`**: Predict article label and extract linguistic characteristics.
  - Request body:
    ```json
    {
      "title": "Article Headline",
      "text": "Full body text...",
      "model": "SVM"
    }
    ```
- **`GET /api/metrics`**: Returns calculated benchmark metrics, training times, and confusion matrices for all 3 models.
- **`GET /api/samples`**: Returns pre-loaded benchmark articles for testing.
