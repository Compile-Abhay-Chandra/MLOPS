import sys
import joblib
import pandas as pd

from breast_cancer.validate import validate_measurements


model_bundle = joblib.load("breast_cancer_model.joblib")
model = model_bundle["model"]
feature_names = model_bundle["feature_names"]

values = [float(value) for value in sys.argv[1:]]

if len(values) != len(feature_names):
    raise ValueError(
        f"Expected {len(feature_names)} measurements, "
        f"but received {len(values)}"
    )

measurements = dict(zip(feature_names, values))
validate_measurements(measurements)

input_data = pd.DataFrame([measurements], columns=feature_names)

prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]
confidence = probabilities.max() * 100

label = {
    "M": "malignant",
    "B": "benign"
}[prediction]

print("Prediction:", label)
print(f"Confidence: {confidence:.2f}%")
