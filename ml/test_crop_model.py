from pathlib import Path

import joblib
import pandas as pd

# Find the project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load the saved model
model_path = PROJECT_DIR / "ml" / "models" / "crop_model.pkl"

saved_data = joblib.load(model_path)

model = saved_data["model"]
label_encoder = saved_data["label_encoder"]
feature_names = saved_data["feature_names"]

print("Saved model loaded successfully!")
print("Model name:", saved_data["model_name"])

# Sample soil and weather values
sample_data = pd.DataFrame(
    [[90, 42, 43, 20.8, 82, 6.5, 202]],
    columns=feature_names
)

# Predict the crop
prediction = model.predict(sample_data)

# Convert encoded prediction to crop name
crop_name = label_encoder.inverse_transform(prediction)

print("\nSoil and Weather Input:")
print(sample_data)

print("\nRecommended Crop:", crop_name[0])
