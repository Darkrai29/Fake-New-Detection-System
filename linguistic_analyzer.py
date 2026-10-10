"""
Linguistic and Textual Characteristics Analyzer
Extracts comprehensive stylistic, syntactic, and lexical metrics
from input articles, providing explainability for fake vs genuine predictions.
"""

import re
import math
from collections import Counter

# Sensationalist / clickbait / hyper-emotional cue words
SENSATIONAL_LEXICON = {
    "shocking", "bombshell", "unbelievable", "miracle", "exposed", "secret",
    "conspiracy", "hoax", "censored", "forbidden", "destroy", "devastating",
    "panic", "horrifying", "scandalous", "wake up", "truth revealed", "deep state",
    "mainstream media", "rigged", "collusion", "traitor", "treason", "alert",
    "urgent", "must see", "viral", "uncovered", "plot", "insane", "nightmare",
    "explosive", "smoking gun", "busted", "apocalypse", "coverup", "cover-up"
}

# Journalistic attribution / objective reporting cues
ATTRIBUTION_LEXICON = {
    "said", "stated", "reported", "according to", "spokesperson", "spokesman",
    "officials", "confirmed", "statement", "sources", "agency", "reuters",
    "associated press", "ap", "announced", "testified", "documented", "briefing",
    "investigation", "department", "representative", "conference", "inquiry"
}

# Personal pronouns (first/second person vs third person journalistic neutrality)
PERSONAL_PRONOUNS = {
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours"
}

def analyze_linguistics(raw_text, title=""):
    """
    Performs full linguistic and textual analysis on the input article.
    Returns a dictionary of stylistic, structural, and affective metrics.
    """
    full_text = f"{title.strip()} {raw_text.strip()}".strip()
    if not full_text:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_len": 0,
            "avg_word_len": 0,
            "lexical_diversity": 0,
            "uppercase_ratio": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "repeated_punct_count": 0,
            "sensationalism_score": 0,
            "attribution_score": 0,
            "personal_pronoun_ratio": 0,
            "sensational_words_found": [],
            "attribution_words_found": [],
            "summary_bullet_points": ["Text is empty."]
        }

    # 1. Structural & Sentence Metrics
    sentences = [s.strip() for s in re.split(r'[.!?]+', full_text) if len(s.strip()) > 0]
    sentence_count = max(1, len(sentences))
    
    words_raw = re.findall(r'\b[A-Za-z]+\b', full_text)
    total_words = max(1, len(words_raw))
    words_lower = [w.lower() for w in words_raw]
    
    avg_sentence_len = round(total_words / sentence_count, 1)
    avg_word_len = round(sum(len(w) for w in words_raw) / total_words, 1)
    
    # 2. Lexical Diversity (Type-Token Ratio)
    unique_words = len(set(words_lower))
    lexical_diversity = round((unique_words / total_words) * 100, 1)
    
    # 3. Capitalization / ALL CAPS Analysis
    # Words in all caps with length >= 2
    all_caps_words = [w for w in words_raw if w.isupper() and len(w) >= 2 and w not in {"US", "UK", "EU", "UN", "FBI", "CIA", "GOP", "DNA", "TV", "CEO", "PM"}]
    uppercase_ratio = round((len(all_caps_words) / total_words) * 100, 2)
    
    # 4. Punctuation Anomalies
    exclamation_count = full_text.count('!')
    question_count = full_text.count('?')
    repeated_punct = len(re.findall(r'[!?]{2,}', full_text))
    
    # 5. Sensationalism & Emotion Marker Detection
    sensational_found = []
    text_lower = full_text.lower()
    for phrase in SENSATIONAL_LEXICON:
        if " " in phrase:
            if phrase in text_lower:
                sensational_found.append(phrase)
        else:
            if phrase in words_lower:
                sensational_found.append(phrase)
                
    sensationalism_density = round((len(sensational_found) / total_words) * 1000, 2)
    # Scaled sensationalism score out of 100
    sensationalism_score = min(100, round((len(sensational_found) * 15) + (uppercase_ratio * 3) + (repeated_punct * 10), 1))
    
    # 6. Journalistic Attribution Markers
    attribution_found = []
    for phrase in ATTRIBUTION_LEXICON:
        if " " in phrase:
            if phrase in text_lower:
                attribution_found.append(phrase)
        else:
            if phrase in words_lower:
                attribution_found.append(phrase)
                
    attribution_density = round((len(attribution_found) / total_words) * 1000, 2)
    attribution_score = min(100, round(len(attribution_found) * 12 + (15 if "said" in words_lower else 0), 1))
    
    # 7. Personal Pronoun Ratio (Informal / Subjective narrative)
    personal_pronoun_count = sum(1 for w in words_lower if w in PERSONAL_PRONOUNS)
    personal_pronoun_ratio = round((personal_pronoun_count / total_words) * 100, 2)
    
    # 8. Human-Readable Bullet Points
    bullets = []
    if (uppercase_ratio > 6.0 and exclamation_count > 0) or repeated_punct > 0 or exclamation_count >= 2:
        bullets.append(f"Elevated emotional punctuation detected ({exclamation_count} exclamations, {len(all_caps_words)} ALL-CAPS words). This is typical of clickbait/sensational styling.")
    else:
        bullets.append("Restrained typographical styling with conventional punctuation, consistent with standard editorial journalism.")
        
    if sensational_found:
        bullets.append(f"Found {len(sensational_found)} hyper-emotive/sensational trigger keywords: {', '.join(sensational_found[:5])}.")
    else:
        bullets.append("No prominent hyper-emotive sensationalism markers detected.")
        
    if attribution_found:
        bullets.append(f"Contains {len(attribution_found)} journalistic attribution markers ({', '.join(attribution_found[:4])}), indicating third-person source reporting.")
    else:
        bullets.append("Sparse journalistic source attribution detected.")
        
    if lexical_diversity > 60:
        bullets.append(f"High vocabulary diversity ({lexical_diversity}% TTR), indicating rich descriptive text.")
    elif lexical_diversity < 35:
        bullets.append(f"Moderate-to-low vocabulary diversity ({lexical_diversity}% TTR), indicating repetitive phrasing.")
        
    return {
        "word_count": total_words,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "avg_word_len": avg_word_len,
        "lexical_diversity": lexical_diversity,
        "uppercase_ratio": uppercase_ratio,
        "all_caps_count": len(all_caps_words),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "repeated_punct_count": repeated_punct,
        "sensationalism_score": sensationalism_score,
        "sensational_words_found": sensational_found[:8],
        "attribution_score": attribution_score,
        "attribution_words_found": attribution_found[:8],
        "personal_pronoun_ratio": personal_pronoun_ratio,
        "summary_bullet_points": bullets
    }

def extract_top_salient_tokens(text, vectorizer, svm_model, top_k=8):
    """
    Extracts the highest-impact TF-IDF tokens present in the text
    and explains whether each token signals Fake or Genuine based on model weights.
    """
    try:
        # Preprocess text
        from train_models import clean_text
        cleaned = clean_text(text)
        tfidf_vec = vectorizer.transform([cleaned])
        
        # Get active feature indices in this text
        feature_indices = tfidf_vec.nonzero()[1]
        if len(feature_indices) == 0:
            return []
            
        feature_names = vectorizer.get_feature_names_out()
        
        # CalibratedClassifierCV wraps LinearSVC estimators
        if hasattr(svm_model, 'calibrated_classifiers_'):
            # Average weights across calibrated folds
            weights = sum(clf.estimator.coef_[0] for clf in svm_model.calibrated_classifiers_) / len(svm_model.calibrated_classifiers_)
        elif hasattr(svm_model, 'coef_'):
            weights = svm_model.coef_[0]
        else:
            # Fallback for trees: use tfidf values
            weights = None
            
        salient_tokens = []
        for idx in feature_indices:
            token = feature_names[idx]
            tfidf_val = tfidf_vec[0, idx]
            weight = float(weights[idx]) if weights is not None else 0.0
            
            # Impact is the product of TF-IDF presence and model coefficient
            impact = float(weight * tfidf_val)
            salient_tokens.append({
                "token": token,
                "tfidf": round(float(tfidf_val), 3),
                "weight": round(float(weight), 3),
                "impact": round(impact, 4),
                "leaning": "Fake News" if weight > 0 else "Genuine News"
            })
            
        # Sort by absolute impact
        salient_tokens.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return salient_tokens[:top_k]
    except Exception as e:
        print(f"Error extracting salient tokens: {e}")
        return []
