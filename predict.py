import sys, joblib, json
from scipy.sparse import hstack
from features import FeatureExtractor

THRESHOLD = 0.5

def load_artifacts():
    clf = joblib.load("models/rf_model.joblib")
    fe = FeatureExtractor(); fe.load("models/tfidf.joblib")
    return clf, fe

def predict(query, clf, fe):
    X_text, X_num = fe.transform([query])
    X = hstack([X_text, X_num.values])
    score = float(clf.predict_proba(X)[0,1]) if hasattr(clf, "predict_proba") else float(clf.predict(X)[0])
    label = "ATTACK" if score >= THRESHOLD else "SAFE"
    return {"query": query, "score": score, "label": label}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"<SQL query>\"")
        sys.exit(1)
    q = sys.argv[1]
    clf, fe = load_artifacts()
    out = predict(q, clf, fe)
    print(json.dumps(out, indent=2))
