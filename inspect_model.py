import joblib, pprint, sys, os

path = "backend/model/model.joblib"
if not os.path.exists(path):
    print("No model file at", path)
    sys.exit(0)

obj = joblib.load(path)
print("Loaded type:", type(obj))

if isinstance(obj, dict):
    print("DICT keys:")
    pprint.pprint(list(obj.keys()))
else:
    attrs = {}
    for a in ("predict","predict_proba","classes_","named_steps"):
        attrs[a] = hasattr(obj,a)
    print("Estimator attrs:", attrs)

    try:
        print("repr:", repr(obj)[:500])
    except Exception as e:
        print("repr failed:", e)
