# backend/app.py
"""
Wealth_Port_AL backend (FastAPI)
- Loads estimator-only model from model_est/model.joblib (relative to backend/)
- Endpoints:
  GET  /          -> basic info
  GET  /health    -> model health
  GET  /users     -> returns sample rows from athena_users_sample.csv (if present)
  POST /predict   -> predict country from {"name": "..."}
"""

from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
import csv
import traceback

# ---------- config ----------
MODEL_PATH = os.getenv("MODEL_PATH", "model_est/model.joblib")  # relative to backend/
SAMPLE_USERS_CSV = os.getenv("SAMPLE_USERS_CSV", "athena_users_sample.csv")
# ---------- /config ----------

app = FastAPI(title="Wealth_Port_AL Backend")

class PredictRequest(BaseModel):
    name: str

# Try to load model (estimator-only or dict containing estimator)
model = None
try:
    import joblib
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}", file=sys.stdout)
except Exception as e:
    print(f"Warning: failed to load model at {MODEL_PATH}: {e}", file=sys.stderr)
    model = None

# Helper: find estimator and label_encoder inside loaded object (dict or estimator)
def extract_estimator_and_le(loaded):
    est = None
    le = None
    try:
        if loaded is None:
            return None, None
        if isinstance(loaded, dict):
            # common keys
            est = loaded.get("model") or loaded.get("estimator") or loaded.get("pipeline") or loaded.get("clf")
            le = loaded.get("le") or loaded.get("label_encoder") or loaded.get("lbl_enc")
            if est is None:
                # fallback: pick first value that looks like estimator
                for v in loaded.values():
                    if hasattr(v, "predict") or hasattr(v, "predict_proba"):
                        est = v
                        break
        else:
            est = loaded
    except Exception:
        est = None
    return est, le

estimator, label_encoder = extract_estimator_and_le(model)

# debug info
try:
    if estimator is not None:
        print("Estimator type:", type(estimator), file=sys.stdout)
        print("Estimator n_features_in_:", getattr(estimator, "n_features_in_", None), file=sys.stdout)
        print("Estimator classes_:", getattr(estimator, "classes_", None), file=sys.stdout)
except Exception:
    pass

# Build feature vector compatibly with estimator expected features
def build_vec_for_model(name: str, estimator_obj):
    import numpy as np
    # determine expected n features
    n_in = getattr(estimator_obj, "n_features_in_", None)
    # If unknown or 1, use single feature (length)
    if n_in is None or n_in == 1:
        return np.array([[len(name)]], dtype=float)
    if n_in == 2:
        return np.array([[len(name), sum(map(ord, name))]], dtype=float)
    # fallback: create deterministic list and pad/trim to required length
    vals = [len(name), sum(map(ord, name)), len(name) % 10, sum(map(ord, name)) % 100]
    while len(vals) < n_in:
        vals.append(vals[-1])
    if len(vals) > n_in:
        vals = vals[:n_in]
    return np.array([vals], dtype=float)

@app.get("/")
def root():
    return {"status": "ok", "service": "Wealth_Port_AL backend"}

@app.get("/health")
def health():
    return {"healthy": True, "model_loaded": bool(estimator)}

@app.get("/users")
def users(limit: int = 50):
    # Return sample rows from committed CSV if present (helpful for demo)
    if os.path.exists(SAMPLE_USERS_CSV):
        rows = []
        try:
            with open(SAMPLE_USERS_CSV, newline="", encoding="utf-8") as fh:
                r = csv.DictReader(fh)
                for i, row in enumerate(r):
                    if i >= limit:
                        break
                    rows.append(row)
            return {"count": len(rows), "rows": rows}
        except Exception as e:
            return {"count": 0, "rows": [], "error": str(e)}
    return {"count": 0, "rows": []}

@app.post("/predict")
def predict(req: PredictRequest):
    name = (req.name or "").strip()
    if name == "":
        return {"error": "empty name provided"}

    est = estimator
    le = label_encoder

    # If estimator not available, return fallback distribution
    if est is None:
        scores = {"IN": 0.6, "UK": 0.2, "US": 0.2}
        return {"country_pred": max(scores, key=scores.get), "scores": scores, "warning": "No usable model; returned fallback."}

    try:
        import numpy as np
        vec = build_vec_for_model(name, est)

        # prefer predict_proba if available
        if hasattr(est, "predict_proba"):
            probs = est.predict_proba(vec)
            if probs.ndim == 2:
                probs = probs[0]
            # get classes
            classes = getattr(est, "classes_", None)
            if classes is None and le is not None and hasattr(le, "classes_"):
                classes = le.classes_
            if classes is not None:
                scores = {str(c): float(p) for c, p in zip(classes, probs)}
            else:
                scores = {str(i): float(p) for i, p in enumerate(probs)}
            best = max(scores, key=scores.get) if scores else None
            return {"country_pred": best, "scores": scores}
        else:
            # only predict available
            pred = est.predict(vec)
            pred0 = pred[0] if hasattr(pred, "__iter__") else pred
            try:
                if le is not None and hasattr(le, "inverse_transform"):
                    label_val = le.inverse_transform([int(pred0)])[0]
                else:
                    label_val = str(pred0)
            except Exception:
                label_val = str(pred0)
            return {"country_pred": label_val, "scores": {}}
    except Exception:
        tb = traceback.format_exc()
        print("PREDICT ERROR:", tb, file=sys.stderr)
        # safe fallback
        scores = {"IN": 0.6, "UK": 0.2, "US": 0.2}
        return {"country_pred": max(scores, key=scores.get), "scores": scores, "warning": "Model error - fallback used."}
