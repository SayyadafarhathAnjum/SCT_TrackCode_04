# ============================================================
# Hand Gesture Recognition Model
# SkillCraft Technology - Task 04
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SETTINGS
# ============================================================
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15

# ============================================================
# STEP 1: Load Dataset
# Dataset folder structure:
# dataset/
#   gesture_01/ (images)
#   gesture_02/ (images)
#   ...
# ============================================================

def load_dataset(base_path):
    data, labels = [], []
    gesture_classes = sorted(os.listdir(base_path))
    
    for gesture in gesture_classes:
        folder = os.path.join(base_path, gesture)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder)[:200]:
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(gesture)
    
    return np.array(data), np.array(labels)

print("Loading dataset...")
X, y = load_dataset('dataset/')

# Normalize
X = X / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"Classes found: {le.classes_}")
print(f"Total images: {len(X)}")

# One-hot encode
y_cat = tf.keras.utils.to_categorical(y_encoded, num_classes)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# STEP 2: Build CNN Model
# ============================================================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu',
                  input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D(2,2),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.BatchNormalization(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# STEP 3: Train Model
# ============================================================
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

# ============================================================
# STEP 4: Evaluate
# ============================================================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*50}")
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred,
      target_names=le.classes_))

# ============================================================
# STEP 5: Plot Results
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()
print("[Saved] training_history.png")

# ============================================================
# STEP 6: Save Model
# ============================================================
model.save('gesture_model.h5')
np.save('gesture_classes.npy', le.classes_)
print("[Saved] gesture_model.h5")
print("\n✅ Task 04 Complete!")