from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# 🔥 rebuild model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ✅ load weights (NOT load_model)
model.load_weights("fruit_model/weights.weights.h5")

# class labels (IMPORTANT — match your training)
class_labels = ["Apple", "Orange"]

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

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
            label = "Orange 🍊"
            confidence = confidence_val
        else:
            label = "Apple 🍎"
            confidence = 1 - confidence_val

        prediction = label
        confidence = round(confidence * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)