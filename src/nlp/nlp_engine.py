import joblib
import json
import os
from .bert_processor import BertProcessor

class NLPEngine:
    def __init__(self):
        # Load ML Models
        self.tfidf_model = joblib.load("./models/nlp/baseline_tfidf.pkl")
        self.bert = BertProcessor()
        self.bert_clf = joblib.load("./models/nlp/expert_bert_svm.pkl")
        
        # Load Keyword Knowledge Base
        with open(os.path.join(os.path.dirname(__file__), "keywords.json"), 'r', encoding='utf-8') as f:
            self.keyword_dict = json.load(f)

    def validate_with_keywords(self, text, predicted_label):
        """Returns a score from 0 to 1 based on keyword matches."""
        text = text.lower()
        must_have = self.keyword_dict.get(predicted_label, [])
        if not must_have:
            return 0.5
        
        matches = [word for word in must_have if word in text]
        return len(matches) / len(must_have)

    def predict(self, raw_text):
        # 1. Baseline Prediction (Fast)
        tfidf_pred = self.tfidf_model.predict([raw_text])[0]
        
        # 2. Expert Prediction (Accurate)
        # We use BERT specifically if it's an ID or if TF-IDF is low-confidence
        if "CIN" in tfidf_pred:
            vector = self.bert.get_embeddings(raw_text).reshape(1, -1)
            final_label = self.bert_clf.predict(vector)[0]
            method = "Expert (BERT)"
        else:
            final_label = tfidf_pred
            method = "Baseline (TF-IDF)"

        # 3. Keyword Validation
        kw_score = self.validate_with_keywords(raw_text, final_label)
        
        # Logic: If ML says 'Facture' but zero keywords match, lower the confidence
        if kw_score > 0.5:
            confidence = 0.98
        elif kw_score > 0.1:
            confidence = 0.85
        else:
            confidence = 0.60 # ML predicted it, but keywords didn't find proof

        return {
            "label": final_label,
            "confidence": confidence,
            "method": method,
            "keyword_proof_score": round(kw_score, 2)
        }