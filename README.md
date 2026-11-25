# ManggaAI

Sistem Klasifikasi Penyakit Buah Mangga menggunakan model Deep Learning VGG16 dengan arsitektur **Backend (Flask)** dan **Frontend (Vite + React + Tailwind CSS)** yang terpisah.

## Struktur Proyek

```text
manggo-agent/
├── backend/
│   ├── model/
│   │   ├── mangga_banget.ipynb
│   │   └── model_vgg16.keras
│   ├── main.py
│   ├── procfile
│   ├── requirements.txt
│   └── runtime.txt
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── ClassifierCard.jsx
    │   │   ├── HistorySection.jsx
    │   │   └── Navbar.jsx
    │   ├── App.jsx
    │   ├── index.css
    │   └── main.jsx
    ├── .env.example
    ├── index.html
    ├── package.json
    └── vite.config.js
```

## Menjalankan Aplikasi

### 1. Backend (Flask API)

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Backend berjalan di `http://127.0.0.1:5000`.

### 2. Frontend (Vite + React)

```bash
cd frontend
npm install
npm run dev
```

Frontend berjalan di `http://localhost:5173` dan otomatis meneruskan request `/classify` ke backend Flask.
