from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import io
import os
from PIL import Image

app = Flask(__name__, template_folder='templates')

# --- Konfigurasi Model ---
MODEL_PATH = 'model/model_vgg16.keras'
IMAGE_SIZE = (224, 224)

CLASS_NAMES = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Healthy', 'Stem And rot']
# Threshold confidence minimal untuk deteksi "bukan buah mangga"
CONFIDENCE_THRESHOLD = 0.9  # 70%

model = None

# --- Load Model ---
def load_and_initialize_model():
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model tidak ditemukan: {MODEL_PATH}")
        return

    try:
        # Jika model menggunakan backbone VGG16, tambahkan custom_objects
        custom_objects = {
            "VGG16": tf.keras.applications.VGG16
        }

        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(MODEL_PATH)

        print("Model berhasil dimuat.")

        # Warmup
        dummy = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
        model.predict(dummy)
        print("Warm-up model sukses.")

    except Exception as e:
        print(f"Gagal memuat model: {e}")
        model = None


load_and_initialize_model()


# --- Preprocess Image ---
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)

    img_array = np.array(img) / 255.0  # Jika training pakai normalisasi 0–1
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    if model is None:
        return jsonify({'error': 'Model gagal dimuat. Cek log server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dikirim.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nama file kosong.'}), 400

    try:
        # Load gambar
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)

        # Prediksi
        preds = model.predict(processed_image)[0]


        best_idx = np.argmax(preds)
        best_label = CLASS_NAMES[best_idx]
        confidence = float(preds[best_idx])

        # Jika confidence di bawah threshold, anggap bukan buah mangga
        if confidence < CONFIDENCE_THRESHOLD:
            best_label = "Bukan buah mangga"

        all_pred = [
            {"label": CLASS_NAMES[i], "p": float(p)}
            for i, p in enumerate(preds)
        ]

        return jsonify({
            "label": best_label,
            "confidence": confidence,
            "predictions": all_pred
        })

    except Exception as e:
        return jsonify({"error": f"Kesalahan saat memproses gambar: {e}"}), 500


if __name__ == '__main__':
    # Untuk development lokal
    app.run(host="0.0.0.0", port=5000)