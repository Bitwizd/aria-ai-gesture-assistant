import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# -----------------------------
# Paths
# -----------------------------
DATASET_FILE = os.path.join("dataset", "hand_gestures.csv")
MODEL_OUTPUT_DIR = "models"
MODEL_OUTPUT_FILE = os.path.join(MODEL_OUTPUT_DIR, "gesture_classifier.pkl")

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Gesture Label Mapping
# -----------------------------
GESTURES = {
    0: "Open Palm",
    1: "Fist",
    2: "Thumbs Up",
    3: "Peace",
    4: "Pointing"
}


# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")

df = pd.read_csv(DATASET_FILE)

print(f"Total samples loaded: {len(df)}")
print()

print("Class distribution:")
label_counts = df["label"].value_counts().sort_index()

for label, count in label_counts.items():
    gesture_name = GESTURES.get(int(label), str(label))
    print(f"  {gesture_name} (label {label}): {count} samples")

print()


# -----------------------------
# Basic Validation
# -----------------------------
expected_columns = 63 + 1  # 21 landmarks * 3 coords + label

if df.shape[1] != expected_columns:
    print(
        f"WARNING: Expected {expected_columns} columns, "
        f"found {df.shape[1]}. Check dataset integrity."
    )

missing_count = df.isnull().sum().sum()

if missing_count > 0:
    print(f"WARNING: Found {missing_count} missing values. Dropping affected rows.")
    df = df.dropna()

print(f"Samples after validation: {len(df)}")
print()


# -----------------------------
# Split Features and Labels
# -----------------------------
X = df.drop(columns=["label"])
y = df["label"].astype(int)


# -----------------------------
# Train/Test Split (Stratified)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print()


# -----------------------------
# Train Model
# -----------------------------
print("Training RandomForestClassifier...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete.")
print()


# -----------------------------
# Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print()

print("Classification Report:")
sorted_labels = sorted(GESTURES.keys())
target_names = [GESTURES[label] for label in sorted_labels]
print(
    classification_report(
        y_test,
        y_pred,
        labels=sorted_labels,
        target_names=target_names,
        zero_division=0
    )
)

print("Confusion Matrix:")
print("Rows = actual, Columns = predicted")
print(f"Order: {target_names}")
cm = confusion_matrix(y_test, y_pred, labels=sorted_labels)
print(cm)
print()


# -----------------------------
# Feature Importance (top 10)
# -----------------------------
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("Top 10 most important features:")
print(feature_importance.head(10))
print()


# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, MODEL_OUTPUT_FILE)

print(f"Model saved to: {MODEL_OUTPUT_FILE}")
