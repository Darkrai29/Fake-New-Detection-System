"""
Fake News Detection - Model Training Pipeline
Trains SVM (LinearSVC with calibration), Random Forest, and XGBoost on the dataset.
Calculates Accuracy, Precision, Recall, F1-score, and Confusion Matrix on the test set.
Saves all models, vectorizer, and computed metrics.
"""

import os
import re
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def clean_text(text):
    """
    Cleans raw text by stripping HTML, URLs, non-alphabetic symbols,
    and converting to lowercase.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # Keep letters, basic punctuation for tokenization
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Normalize whitespaces and lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def train_and_evaluate():
    print("=" * 60)
    print("FAKENEWS DETECTION - MODEL TRAINING & BENCHMARKING")
    print("=" * 60)
    
    # 1. Load Dataset
    data_path = 'news_dataset.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Initial dataset shape: {df.shape}")
    
    # 2. Prepare Data
    # Drop rows with missing text or title
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    # Combine title and text for rich linguistic context
    df['combined_text'] = df['title'] + " " + df['text']
    df = df[df['combined_text'].str.strip() != '']
    
    # Clean label: FAKE -> 1, REAL -> 0
    df['label_binary'] = df['label'].apply(lambda x: 1 if str(x).strip().upper() == 'FAKE' else 0)
    
    print(f"Processed valid samples: {len(df)}")
    print(f"Label balance: Fake={sum(df['label_binary']==1)}, Genuine={sum(df['label_binary']==0)}")
    
    # 3. Text Preprocessing
    print("Preprocessing text (cleaning HTML, URLs, punctuation)...")
    t0 = time.time()
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    print(f"Text cleaning completed in {time.time() - t0:.2f}s")
    
    # 4. Stratified Train / Test Split (80% train, 20% test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['label_binary'],
        test_size=0.20,
        random_state=42,
        stratify=df['label_binary']
    )
    print(f"Train size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")
    
    # 5. TF-IDF Feature Extraction
    print("Fitting TF-IDF Vectorizer...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print(f"TF-IDF feature extraction done in {time.time() - t0:.2f}s. Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # 6. Initialize Models
    # SVM: LinearSVC calibrated for accurate probability estimates
    base_svm = LinearSVC(C=1.0, random_state=42, max_iter=3000)
    svm_model = CalibratedClassifierCV(estimator=base_svm, cv=3)
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=30,
        random_state=42,
        n_jobs=-1
    )
    
    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.15,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    models = {
        'SVM': {
            'name': 'Support Vector Machine (Linear)',
            'model': svm_model
        },
        'Random Forest': {
            'name': 'Random Forest Classifier',
            'model': rf_model
        },
        'XGBoost': {
            'name': 'XGBoost Classifier',
            'model': xgb_model
        }
    }
    
    results = {}
    best_model_key = None
    best_f1 = -1.0
    
    # 7. Train & Evaluate Models
    for key, info in models.items():
        print(f"\n--- Training {info['name']} ({key}) ---")
        t_start = time.time()
        clf = info['model']
        clf.fit(X_train_tfidf, y_train)
        train_time = time.time() - t_start
        
        # Predictions on Test Set
        y_pred = clf.predict(X_test_tfidf)
        
        # Calculate All Required Metrics directly from test split
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, pos_label=1))
        rec = float(recall_score(y_test, y_pred, pos_label=1))
        f1 = float(f1_score(y_test, y_pred, pos_label=1))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Confusion matrix elements: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        print(f"Results for {key}:")
        print(f"  Training Time: {train_time:.2f}s")
        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall:    {rec * 100:.2f}%")
        print(f"  F1-Score:  {f1 * 100:.2f}%")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Save model artifact
        model_filename = f"models/{key.lower().replace(' ', '_')}.joblib"
        joblib.dump(clf, model_filename)
        
        results[key] = {
            'key': key,
            'name': info['name'],
            'model_file': model_filename,
            'train_time_sec': round(train_time, 2),
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'confusion_matrix': {
                'matrix': cm,
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'test_samples': len(y_test)
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_key = key
            
    print("\n" + "=" * 60)
    print(f"BEST PERFORMING MODEL: {best_model_key} (F1-Score: {best_f1 * 100:.2f}%)")
    print("=" * 60)
    
    # Save vectorizer
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    
    # Save a set of test examples for instant interactive testing in UI
    # Pick 2 genuine and 2 fake test samples
    test_df = df.iloc[y_test.index].copy()
    test_samples = []
    fake_samples = test_df[test_df['label_binary'] == 1].head(4)
    real_samples = test_df[test_df['label_binary'] == 0].head(4)
    
    for _, row in pd.concat([fake_samples, real_samples]).iterrows():
        test_samples.append({
            'title': row['title'],
            'text': row['text'][:1500], # excerpt
            'ground_truth': 'FAKE NEWS' if row['label_binary'] == 1 else 'GENUINE NEWS'
        })
    
    metadata = {
        'total_dataset_samples': len(df),
        'train_samples': len(X_train_raw),
        'test_samples': len(X_test_raw),
        'tfidf_features': len(vectorizer.vocabulary_),
        'best_model': best_model_key,
        'models': results,
        'sample_articles': test_samples,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Training artifacts and metrics saved successfully to 'models/' directory.")
    return metadata

if __name__ == '__main__':
    train_and_evaluate()
