from pathlib import Path

import joblib
import pandas as pd

# Locate the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load the saved crop recommendation model
model_path = PROJECT_DIR / "ml" / "models" / "crop_model.pkl"

saved_data = joblib.load(model_path)

model = saved_data["model"]
label_encoder = saved_data["label_encoder"]
feature_names = saved_data["feature_names"]

print("Crop prediction model loaded successfully!")
print("Model:", saved_data["model_name"])

# Enter soil and weather values
N = 90
P = 42
K = 43
Temperature = 20.8
Humidity = 82
ph = 6.5
Rainfall = 202

# Prepare the input in the correct feature order
input_data = pd.DataFrame(
    [[N, P, K, Temperature, Humidity, ph, Rainfall]],
    columns=feature_names
)

# Predict the crop
prediction = model.predict(input_data)

# Convert the prediction into a crop name
recommended_crop = label_encoder.inverse_transform(prediction)[0]

# Display the result
print("\n========== CROP PREDICTION ==========")
print("N:", N)
print("P:", P)
print("K:", K)
print("Temperature:", Temperature)
print("Humidity:", Humidity)
print("pH:", ph)
print("Rainfall:", Rainfall)
print("-------------------------------------")
print("Recommended Crop:", recommended_crop.capitalize())
print("=====================================")
