from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib, time, os, json
from scipy.sparse import hstack
from features import FeatureExtractor

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load model artifacts (will raise if missing)
MODEL_PATH = "models/rf_model.joblib"
TFIDF_PATH = "models/tfidf.joblib"
if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
    raise FileNotFoundError("Model artifacts not found. Run train_model.py first to create models/")

clf = joblib.load(MODEL_PATH)
fe = FeatureExtractor(); fe.load(TFIDF_PATH)

LOGFILE = "logs/alerts.jsonl"
os.makedirs("logs", exist_ok=True)

THRESH = 0.5

@app.route("/")
def index():
    # serves templates/index.html
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    q = data.get("query", "")
    timestamp = time.time()

    X_text, X_num = fe.transform([q])
    X = hstack([X_text, X_num.values])
    try:
        score = float(clf.predict_proba(X)[0,1]) if hasattr(clf, "predict_proba") else float(clf.predict(X)[0])
    except Exception as e:
        return jsonify({"error": "Model inference failed", "detail": str(e)}), 500

    action = "BLOCK" if score >= THRESH else "ALLOW"
    label = "ATTACK" if action == "BLOCK" else "SAFE"

    rec = {"ts": timestamp, "query": q, "score": score, "label": label, "action": action}
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    return jsonify(rec)

# Optional: allow fetching static files if needed
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    # Use host=0.0.0.0 if you want to access from other devices on the network
    app.run(port=5000, debug=False)
