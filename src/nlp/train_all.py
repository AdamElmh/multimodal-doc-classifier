import pandas as pd
import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from bert_processor import BertProcessor
import tqdm

def train():
    df = pd.read_csv("data/processed_text.csv").dropna()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )

    # 1. TRAIN BASELINE (TF-IDF)
    tfidf_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=2000)),
        ('clf', LinearSVC(class_weight='balanced'))
    ])
    tfidf_pipe.fit(X_train_raw, y_train)
    joblib.dump(tfidf_pipe, "./models/nlp/baseline_tfidf.pkl")
    print("\n--- BASELINE TF-IDF REPORT ---")
    print(classification_report(y_test, tfidf_pipe.predict(X_test_raw)))

    # 2. TRAIN EXPERT (BERT)
    bert = BertProcessor()
    print("\nGenerating BERT Embeddings (CPU)...")
    X_train_bert = np.array([bert.get_embeddings(str(t)) for t in tqdm.tqdm(X_train_raw)])
    X_test_bert = np.array([bert.get_embeddings(str(t)) for t in tqdm.tqdm(X_test_raw)])
    
    bert_clf = LinearSVC(class_weight='balanced', C=0.5)
    bert_clf.fit(X_train_bert, y_train)
    joblib.dump(bert_clf, "./models/nlp/expert_bert_svm.pkl")
    print("\n--- EXPERT BERT REPORT ---")
    print(classification_report(y_test, bert_clf.predict(X_test_bert)))

if __name__ == "__main__":
    train()