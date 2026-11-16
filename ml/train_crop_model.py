
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT / "datasets" / "Crop_recommendation.csv"
)

MODEL_DIR = PROJECT_ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# ==========================================
# STEP 2: CLEAN DATA
# ==========================================

df.columns = df.columns.str.strip()

df = df.drop_duplicates().dropna().copy()

print("Dataset shape after cleaning:", df.shape)


# ==========================================
# STEP 3: SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["label"])
y = df["label"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nNumber of crop classes:", y.nunique())


# ==========================================
# STEP 4: ENCODE CROP LABELS
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nCrop classes:")
print(encoder.classes_)


# ==========================================
# STEP 5: SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# STEP 6: DEFINE FIVE MODELS
# ==========================================

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf")
    ),

    "Naive Bayes": GaussianNB()
}


# ==========================================
# STEP 7: TRAIN AND EVALUATE MODELS
# ==========================================

results = []
trained_models = {}

best_model_name = None
best_model = None
best_f1_score = -1

for name, model in models.items():

    print("\n" + "=" * 45)
    print("Training:", name)

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))

    # Save results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    trained_models[name] = model

    # Select best model using weighted F1-score
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model_name = name
        best_model = model


# ==========================================
# STEP 8: COMPARE ALL MODELS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-score",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False, float_format="%.4f"))


# Save comparison table
results_path = MODEL_DIR / "model_comparison.csv"

results_df.to_csv(results_path, index=False)


# ==========================================
# STEP 9: DISPLAY CLASSIFICATION REPORT
# ==========================================

print("\nBest model:", best_model_name)

best_predictions = best_model.predict(X_test)

print("\nClassification report for the best model:")

print(
    classification_report(
        y_test,
        best_predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)


# ==========================================
# STEP 10: SAVE BEST MODEL
# ==========================================

model_path = MODEL_DIR / "crop_model.pkl"

joblib.dump(
    {
        "model": best_model,
        "label_encoder": encoder,
        "feature_names": X.columns.tolist(),
        "model_name": best_model_name
    },
    model_path
)

print("\nSelected model:", best_model_name)
print("Model saved successfully at:", model_path)


print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("Best model:", best_model_name)
print("Best weighted F1-score:", round(best_f1_score, 4))
print("Saved model to:", model_path)
print("Saved comparison to:", results_path)
