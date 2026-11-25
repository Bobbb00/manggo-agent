import { useState } from 'react'
import Navbar from './components/Navbar'
import ClassifierCard from './components/ClassifierCard'
import HistorySection from './components/HistorySection'

export default function App() {
  const [history, setHistory] = useState([])

  const handleNewResult = (item) => {
    setHistory((prev) => [item, ...prev])
  }

  const handleDeleteHistory = (id) => {
    setHistory((prev) => prev.filter((item) => item.id !== id))
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <section className="flex flex-col justify-center items-center text-center py-14 px-4 bg-white shadow-sm">
        <h2 className="text-3xl font-semibold text-green-700 mb-3">
          Sistem Klasifikasi Penyakit Buah Mangga
        </h2>
        <p className="text-gray-600 max-w-xl mb-6">
          Deteksi otomatis penyakit pada buah mangga menggunakan model deep learning VGG16.
        </p>
        <a
          href="#klasifikasi"
          className="bg-green-600 text-white px-6 py-2.5 rounded-lg hover:bg-green-700 transition font-medium"
        >
          Mulai Klasifikasi
        </a>
      </section>

      <ClassifierCard onNewResult={handleNewResult} />

      <HistorySection history={history} onDelete={handleDeleteHistory} />

      <footer className="bg-green-700 text-white text-center py-4 mt-auto">
        <p className="text-sm">© 2025 Sistem Klasifikasi Mangga - Dibuat untuk penelitian akademik</p>
      </footer>
    </div>
  )
}
