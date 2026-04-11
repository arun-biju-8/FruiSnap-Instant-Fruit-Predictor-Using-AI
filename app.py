```python
from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# =========================
# DOWNLOAD MODEL (if not exists)
# =========================
MODEL_PATH = "weights.weights.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1z2Bf6hPvq-nR3GeRLeyr3GN-PSo-JDUE"
    print("Downloading model...")
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Download complete ✅")

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# BUILD MODEL ARCHITECTURE
# =========================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# =========================
# LOAD WEIGHTS
# =========================
model.load_weights(MODEL_PATH)
print("Model loaded successfully ✅")

# =========================
# CLASS LABELS
# =========================
class_labels = ["Apple 🍎", "Orange 🍊"]

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(file)

        processed = preprocess(img)
        pred = model.predict(processed)

        confidence_val = float(pred[0][0])

        if confidence_val > 0.5:
            label = class_labels[1]
            confidence = confidence_val
        else:
            label = class_labels[0]
            confidence = 1 - confidence_val

        prediction = label
        confidence = round(confidence * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=port)
```
