import os
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# Define your cube colors and training folder
cube_colors = ["green", "white", "red", "orange", "blue", "yellow"]
training_folder = "training_data"

# Color mapping for numeric labels
COLOR_MAP = {
    "green": 0,   # F
    "white": 1,   # U
    "red": 2,     # R
    "orange": 3,  # L
    "blue": 4,    # B
    "yellow": 5   # D
}

print("="*60)
print("TRAINING RUBIK'S CUBE COLOR CLASSIFIER")
print("="*60)

# Load all color files dynamically
frames = []
for color in cube_colors:
    file_path = os.path.join(training_folder, f"{color}.xlsx")
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        # Use numeric labels for consistency
        df["label"] = COLOR_MAP[color]
        frames.append(df)
        print(f"✅ Loaded {len(df)} samples for {color}")
        
        # Show color statistics
        print(f"   Average RGB: ({df['R'].mean():.1f}, {df['G'].mean():.1f}, {df['B'].mean():.1f})")
    else:
        print(f"⚠️ Warning: Missing file {file_path}")

# Merge and shuffle all data
if not frames:
    raise FileNotFoundError("No training data found in 'training_data' folder!")

result = pd.concat(frames).sample(frac=1, random_state=42)
print(f"\n📊 Total samples: {len(result)}")

# CRITICAL FIX: Keep data in original BGR format (as collected)
# OpenCV captures in BGR, so we train on BGR values directly
X = result[["B", "G", "R"]].values  # Changed order to match OpenCV's BGR
Y = result["label"].values

# Split data
test_size = 0.30
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, test_size=test_size, random_state=seed, stratify=Y
)

print(f"✅ Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Normalize the data for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with optimized parameters
print("\n🔄 Training model...")
model = LogisticRegression(
    max_iter=5000, 
    C=1.0,  # Lower C for better generalization
    solver='lbfgs', 
    multi_class='multinomial',
    random_state=42
)
model.fit(X_train_scaled, Y_train)

# Save both model and scaler
model_data = {
    'model': model,
    'scaler': scaler
}
filename = "rgb_model.sav"
pickle.dump(model_data, open(filename, "wb"))
print(f"💾 Model and scaler saved as {filename}")

# Test the model
accuracy = model.score(X_test_scaled, Y_test)
print(f"\n🎯 Model accuracy: {accuracy * 100:.2f}%")

# Show per-color accuracy
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test_scaled)
color_names = ["green", "white", "red", "orange", "blue", "yellow"]

print("\n📈 Per-color performance:")
print(classification_report(Y_test, y_pred, target_names=color_names, digits=3))

print("\n🔍 Confusion Matrix:")
cm = confusion_matrix(Y_test, y_pred)
print("     ", "  ".join([c[:3].upper() for c in color_names]))
for i, row in enumerate(cm):
    print(f"{color_names[i][:3].upper()}: {row}")

# Show which colors are most confused
print("\n⚠️ Most common misclassifications:")
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i][j] > 0:
            print(f"  {color_names[i]} → {color_names[j]}: {cm[i][j]} times")

print("\n" + "="*60)
print("✅ Training complete! Now you can run your detection script.")
print("="*60)