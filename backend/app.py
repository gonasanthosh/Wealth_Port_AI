# backend/app.py

from fastapi import FastAPI, HTTPException
import os, time
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import joblib
import numpy as np
from fastapi import Body


# Load .env file
load_dotenv()

app = FastAPI(title="WealthPort API")

# Environment variables
REGION = os.getenv("AWS_REGION", "ap-south-2")
ATHENA_DB = os.getenv("ATHENA_DB", "wealthport_athena_db")
BUCKET = os.getenv("BUCKET")
ATHENA_OUTPUT = os.getenv("ATHENA_OUTPUT", f"s3://{BUCKET}/athena-results/")

# Athena client
athena = boto3.client("athena", region_name=REGION)


def run_athena_query(sql: str):
    """Run Athena SQL and return rows as JSON dictionaries."""
    try:
        # Start Athena query
        resp = athena.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": ATHENA_DB},
            ResultConfiguration={"OutputLocation": ATHENA_OUTPUT},
        )
        qid = resp["QueryExecutionId"]

        # Poll until query completes
        while True:
            status = athena.get_query_execution(QueryExecutionId=qid)
            state = status["QueryExecution"]["Status"]["State"]

            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break

            time.sleep(1)

        if state != "SUCCEEDED":
            raise Exception(f"Athena query failed: {status}")

        # Retrieve rows
        results = athena.get_query_results(QueryExecutionId=qid)

        # Extract header
        header = [c["Label"] for c in results["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]

        data = []
        for row in results["ResultSet"]["Rows"][1:]:
            values = [col.get("VarCharValue", "") for col in row["Data"]]
            data.append(dict(zip(header, values)))

        return data

    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/users")
def get_users(limit: int = 100):
    sql = f"SELECT * FROM {ATHENA_DB}.users_processed LIMIT {limit};"
    rows = run_athena_query(sql)
    return {"count": len(rows), "rows": rows}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.joblib")
_model_bundle = None
if os.path.exists(MODEL_PATH):
    try:
        _model_bundle = joblib.load(MODEL_PATH)
    except Exception:
        _model_bundle = None

@app.post("/predict")
def predict(payload: dict = Body(...)):
    """
    Expect JSON: {"name": "Alice"}
    Returns: {"country_pred": "<country>", "scores": {...}}
    """
    if _model_bundle is None:
        raise HTTPException(status_code=500, detail="Model not available. Run train_model.py to create model.joblib")

    model = _model_bundle['model']
    le = _model_bundle['le']
    name = payload.get("name", "")
    name_len = len(name)
    X = np.array([[name_len]])
    pred_idx = model.predict(X)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(X)[0].tolist()
    # map classes->probs
    class_labels = le.inverse_transform(list(range(len(le.classes_))))
    scores = dict(zip(class_labels, [float(x) for x in probs]))
    return {"country_pred": pred_label, "scores": scores}