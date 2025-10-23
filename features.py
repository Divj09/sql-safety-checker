import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

RE_KEYWORDS = r"\b(select|insert|update|delete|drop|union|exec|sleep|benchmark)\b"

def normalize(q: str) -> str:
    if not q: return ""
    q = re.sub(r"/\*.*?\*/", " ", q, flags=re.DOTALL)
    q = re.sub(r"--.*", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip().lower()

def numeric_features(q: str):
    qn = normalize(q)
    return {
        "n_chars": len(qn),
        "n_tokens": len(qn.split()),
        "n_keywords": len(re.findall(RE_KEYWORDS, qn, flags=re.IGNORECASE)),
        "n_quotes": qn.count("'"),
        "n_semicolons": qn.count(";"),
        "n_parentheses": qn.count("(") + qn.count(")")
    }

class FeatureExtractor:
    def __init__(self, tfidf=None):
        self.tfidf = tfidf or TfidfVectorizer(analyzer='char_wb', ngram_range=(3,6), max_features=1500)

    def fit(self, texts):
        norm = [normalize(t) for t in texts]
        self.tfidf.fit(norm)
        return self

    def transform(self, texts):
        norm = [normalize(t) for t in texts]
        X_text = self.tfidf.transform(norm)
        X_num = pd.DataFrame([numeric_features(t) for t in norm])
        return X_text, X_num

    def fit_transform(self, texts):
        norm = [normalize(t) for t in texts]
        X_text = self.tfidf.fit_transform(norm)
        X_num = pd.DataFrame([numeric_features(t) for t in norm])
        return X_text, X_num

    def save(self, path):
        joblib.dump(self.tfidf, path)

    def load(self, path):
        self.tfidf = joblib.load(path)
