import pandas as pd
import joblib, os
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from features import FeatureExtractor

def main():
    df = pd.read_csv("data/queries_labeled.csv")
    X_texts = df["query_text"].fillna("").tolist()
    y = df["label"].astype(int).values

    fe = FeatureExtractor()
    X_text, X_num = fe.fit_transform(X_texts)

    X = hstack([X_text, X_num.values])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=150, class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None

    print("Report:\n", classification_report(y_test, y_pred, digits=4))
    if auc: print("AUC:", auc)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/rf_model.joblib")
    fe.save("models/tfidf.joblib")
    joblib.dump(list(X_num.columns), "models/feature_names.joblib")
    print("Saved model and vectorizer in models/")

if __name__ == "__main__":
    main()
