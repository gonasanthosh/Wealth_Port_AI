# backend/train_model.py
# Train a tiny dummy model using sklearn and save as joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# read sample from local athena csv (or S3 copy)
csv_path = os.path.join(os.path.dirname(__file__), "..", "athena_users_sample.csv")
if not os.path.exists(csv_path):
    # Try current dir fallback
    csv_path = os.path.join(os.path.dirname(__file__), "athena_users_sample.csv")

df = pd.read_csv(csv_path)

# simple target: predict country from name length (toy example)
df['name_len'] = df['name'].str.len()
X = df[['name_len']]
le = LabelEncoder()
y = le.fit_transform(df['country'])

model = LogisticRegression()
model.fit(X, y)

os.makedirs(os.path.join(os.path.dirname(__file__), "model"), exist_ok=True)
joblib.dump({'model': model, 'le': le}, os.path.join(os.path.dirname(__file__), "model", "model.joblib"))
print("Saved model to backend/model/model.joblib")
