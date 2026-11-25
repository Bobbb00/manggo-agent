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
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ClassifierCard.jsx
│   │   │   ├── HistorySection.jsx
│   │   │   └── Navbar.jsx
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── .dockerignore
├── .gitignore
└── Dockerfile
```

## Menjalankan Aplikasi (Development Lokal)

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

## Deploy ke Railway (1 Service Monorepo)

Proyek ini sudah dilengkapi `Dockerfile` multi-stage di root untuk mem-build `frontend/` dan menjalankannya bersama `backend/` dalam satu service:

1. Push repository ke GitHub.
2. Di Railway: **New Project → Deploy from GitHub repo**.
3. Biarkan **Root Directory** tetap di root (`/`) — Railway otomatis mendeteksi `Dockerfile`.
4. Setelah deploy selesai, klik **Settings → Networking → Generate Domain**.
5. Endpoint API `/health` dan `/classify` serta UI React akan berjalan di domain yang sama tanpa perlu mengatur `VITE_API_URL`.
