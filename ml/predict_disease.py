import tensorflow as tf
import numpy as np
import json
import sys

from pathlib import Path
from tensorflow.keras.utils import load_img, img_to_array


# --------------------------------
# 1. Define project paths
# --------------------------------

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent

MODEL_PATH = ML_DIR / "models" / "disease_model.keras"
CLASS_NAMES_PATH = ML_DIR / "models" / "disease_classes.json"

IMG_SIZE = (224, 224)


# --------------------------------
# 2. Disease name mapping
# --------------------------------

disease_names = {
    "healthy": "Tomato Healthy",
    "early_blight": "Tomato Early Blight",
    "late_blight": "Tomato Late Blight",
    "leaf_mold": "Tomato Leaf Mold",
    "septoria_leaf_spot": "Tomato Septoria Leaf Spot"
}


# --------------------------------
# 3. Load trained model
# --------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Disease model not found: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    },
    safe_mode=False,
    compile=False
)

print("Disease detection model loaded successfully!")


# --------------------------------
# 4. Load class names
# --------------------------------

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)


# --------------------------------
# 5. Disease prediction function
# --------------------------------

def predict_disease(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load image and resize
    image = load_img(
        image_path,
        target_size=IMG_SIZE
    )

    # Convert image into array
    image_array = img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Predict disease
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Get highest probability
    predicted_index = np.argmax(predictions[0])

    confidence = float(
        predictions[0][predicted_index] * 100
    )

    predicted_class = class_names[predicted_index]

    disease = disease_names.get(
        predicted_class,
        predicted_class
    )

    return disease, confidence


# --------------------------------
# 6. Run independently
# --------------------------------

if __name__ == "__main__":

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = PROJECT_DIR / "leaf.jpg"

    disease, confidence = predict_disease(image_path)

    print("\n------------------------------")
    print(" PLANT DISEASE DETECTION")
    print("------------------------------")

    print("Disease:", disease)
    print("Confidence:", f"{confidence:.2f}%")

    print("------------------------------")
