from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# MODEL PATH
# =========================
MODEL_PATH = "weights.weights.h5"

# =========================
# DOWNLOAD MODEL (if not exists)
# =========================
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1z2Bf6hPvq-nR3GeRLeyr3GN-PSo-JDUE"
    print("Downloading model...")
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Download complete ✅")
else:
    print("Model already exists ✅")

# =========================
# LAZY LOAD MODEL
# =========================
model = None

def get_model():
    global model
    if model is None:
        print("Loading model...")

        model_local = tf.keras.Sequential([
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

        model_local.load_weights(MODEL_PATH)
        print("Model loaded successfully ✅")

        model = model_local

    return model

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

        model = get_model()  # ✅ lazy load here
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

@app.route("/about")
def about():
    return render_template("about.html")


# =========================
# RUN APP (LOCAL ONLY)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=port)


