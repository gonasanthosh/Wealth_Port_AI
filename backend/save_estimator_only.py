"""
Save only the estimator part of model.joblib
Run from inside backend/ folder:
    python save_estimator_only.py
"""

import joblib
import os

# Because script is executed inside backend/, paths must be relative to this folder
SOURCE_PATH = os.path.join("model", "model.joblib")
TARGET_DIR = os.path.join("model_est")
TARGET_PATH = os.path.join(TARGET_DIR, "model.joblib")

def main():
    print(f"Looking for model at: {SOURCE_PATH}")

    if not os.path.exists(SOURCE_PATH):
        raise FileNotFoundError(f"Source model not found: {SOURCE_PATH}")

    print("Loading model...")
    data = joblib.load(SOURCE_PATH)

    if isinstance(data, dict) and "model" in data:
        estimator = data["model"]
        print("Extracted estimator from saved dict.")
    else:
        raise ValueError("Invalid model format. Expected dict with key 'model'.")

    os.makedirs(TARGET_DIR, exist_ok=True)

    print(f"Saving estimator-only model → {TARGET_PATH}")
    joblib.dump(estimator, TARGET_PATH)

    print("SUCCESS — estimator-only model saved.")

if __name__ == "__main__":
    main()
