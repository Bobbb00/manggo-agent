import io
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# --- Konfigurasi Model ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_vgg16.keras")
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Alternaria",
    "Anthracnose",
    "Black Mould Rot",
    "Healthy",
    "Stem And rot",
]
# Threshold confidence minimal untuk deteksi "bukan buah mangga"
CONFIDENCE_THRESHOLD = 0.9  # 90%

model = None


def load_and_initialize_model():
    """Memuat model VGG16 dari disk dan melakukan warm-up."""
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model tidak ditemukan: {MODEL_PATH}")
        return

    try:
        custom_objects = {"VGG16": tf.keras.applications.VGG16}
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(MODEL_PATH)

        print("Model berhasil dimuat.")

        # Warm-up model dengan dummy input
        dummy = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
        model.predict(dummy)
        print("Warm-up model sukses.")

    except Exception as e:
        print(f"Gagal memuat model: {e}")
        model = None


load_and_initialize_model()


def preprocess_image(image_bytes):
    """Mengubah bytes gambar menjadi tensor ter-normalisasi (1, 224, 224, 3)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)

    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# --- Routes ---
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "ManggaAI Classification API is running",
        "model_loaded": model is not None,
    })


@app.route("/classify", methods=["POST"])
def classify():
    if model is None:
        return jsonify({"error": "Model gagal dimuat. Cek log server."}), 500

    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)

        preds = model.predict(processed_image)[0]
        best_idx = int(np.argmax(preds))
        best_label = CLASS_NAMES[best_idx]
        confidence = float(preds[best_idx])

        # Jika confidence di bawah threshold, anggap bukan buah mangga
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                "is_mango": False,
                "label": "Bukan buah mangga",
                "confidence": None,
                "predictions": [],
            })

        all_pred = [
            {"label": CLASS_NAMES[i], "p": float(p)}
            for i, p in enumerate(preds)
        ]

        return jsonify({
            "is_mango": True,
            "label": best_label,
            "confidence": confidence,
            "predictions": all_pred,
        })

    except Exception as e:
        return jsonify({"error": f"Kesalahan saat memproses gambar: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
