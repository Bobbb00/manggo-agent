from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import io
import os

app = Flask(__name__, template_folder='templates')

# --- Konfigurasi Model ---
# PASTIKAN PATH MODEL INI BENAR
MODEL_PATH = 'model/model_mangga_VGG16.keras'
IMAGE_SIZE = (224, 224) 
# GANTI DENGAN NAMA KELAS PENYAKITMU SESUAI URUTAN SAAT TRAINING
CLASS_NAMES = ['Anthracnose', 'Alternaria', 'Stem end Rot', 'Black Mould Rot', 'Healthy'] 

model = None
# Muat model sekali saat aplikasi dijalankan
model = tf.keras.models.load_model(MODEL_PATH)
try:
    if os.path.exists(MODEL_PATH):
        # Menggunakan tf.keras.models.load_model untuk memuat model .keras
        model = tf.keras.models.load_model(MODEL_PATH) 
        print("Model VGG16 (.keras) berhasil dimuat.")
        # Jalankan prediksi dummy untuk pemanasan
        model.predict(np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    else:
        print(f"ERROR: File model tidak ditemukan di {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Gagal memuat model. Error: {e}")
    
# --- Fungsi Pra-pemrosesan Gambar ---
def preprocess_image(image_bytes):
    """Membaca bytes gambar, mengubah ukuran, dan menormalisasi untuk VGG16."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    
    # Normalisasi untuk VGG16
    img_array = img_array / 255.0
    
    # Tambahkan dimensi batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

# --- Rute Web ---

@app.route('/')
def index():
    # Menampilkan file index.html dari folder templates
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if model is None:
        return jsonify({'error': 'Model belum dimuat di server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang terkirim.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong.'}), 400

    try:
        image_bytes = file.read()
        
        # 1. Pra-pemrosesan
        processed_img = preprocess_image(image_bytes)
        
        # 2. Klasifikasi
        predictions = model.predict(processed_img)[0]
        
        # 3. Ambil hasil terbaik
        predicted_index = np.argmax(predictions)
        predicted_label = CLASS_NAMES[predicted_index]
        confidence_score = float(predictions[predicted_index])
        
        # Siapkan semua prediksi
        all_predictions = []
        for i, p in enumerate(predictions):
            all_predictions.append({
                'label': CLASS_NAMES[i],
                'p': float(p)
            })
            
        # 4. Kirim hasil
        return jsonify({
            'label': predicted_label,
            'confidence': confidence_score,
            'predictions': all_predictions
        })

    except Exception as e:
        print(f"Kesalahan saat klasifikasi: {e}")
        return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {e}'}), 500

if __name__ == '__main__':
    # Jalankan server Flask
    app.run(debug=True)