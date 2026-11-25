import { useState } from 'react'
import { Trash2, ChevronDown, ChevronUp } from 'lucide-react'

function HistoryItem({ item, index, onDelete }) {
  const [showDetail, setShowDetail] = useState(false)

  return (
    <div className="border border-gray-200 rounded-lg p-3.5 bg-white">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          {item.imageSrc && (
            <img
              src={item.imageSrc}
              alt="thumb"
              className="w-12 h-12 object-cover rounded-md border border-gray-200"
            />
          )}
          <div className="flex flex-col">
            <span className="font-medium text-gray-800">
              {index + 1}. {item.label}
            </span>
            <span className="text-xs text-gray-500">{item.waktu}</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-green-700 font-semibold mr-2">{item.akurasi}</span>
          <button
            type="button"
            onClick={() => setShowDetail((prev) => !prev)}
            className="inline-flex items-center gap-1 text-xs md:text-sm bg-blue-600 text-white px-3 py-1.5 rounded-md hover:bg-blue-700 transition cursor-pointer"
          >
            {showDetail ? (
              <>
                <ChevronUp className="w-3.5 h-3.5" />
                Tutup Detail
              </>
            ) : (
              <>
                <ChevronDown className="w-3.5 h-3.5" />
                Lihat Detail
              </>
            )}
          </button>
          <button
            type="button"
            onClick={() => onDelete(item.id)}
            className="inline-flex items-center gap-1 text-xs md:text-sm bg-red-500 text-white px-3 py-1.5 rounded-md hover:bg-red-600 transition cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Hapus
          </button>
        </div>
      </div>

      {showDetail && (
        <div className="mt-4 pt-3 border-t border-gray-100 flex flex-col md:flex-row items-start md:items-center gap-4">
          <div className="shrink-0">
            {item.imageSrc ? (
              <img
                src={item.imageSrc}
                alt="Detail gambar"
                className="w-36 h-36 object-cover rounded-md border border-gray-200"
              />
            ) : (
              <div className="w-36 h-36 bg-gray-100 flex items-center justify-center text-sm text-gray-500 rounded-md">
                No Image
              </div>
            )}
          </div>
          <div className="flex-1 text-sm text-gray-700 space-y-1.5">
            <p>
              <strong>Label:</strong> {item.label}
            </p>
            <p>
              <strong>Akurasi:</strong> {item.akurasi}
            </p>
            <p>
              <strong>Waktu:</strong> {item.waktu}
            </p>
            <div>
              <strong>Prediksi:</strong>
              <ul className="list-disc list-inside text-gray-600 mt-1 space-y-0.5">
                {item.predictions?.length > 0 ? (
                  item.predictions.map((p) => (
                    <li key={p.label}>
                      {p.label}: {(p.p * 100).toFixed(2)}%
                    </li>
                  ))
                ) : (
                  <li>-</li>
                )}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default function HistorySection({ history, onDelete }) {
  return (
    <section id="riwayat" className="container mx-auto pb-12 px-4">
      <div className="bg-white shadow-md rounded-2xl p-6 max-w-3xl mx-auto">
        <h3 className="text-2xl font-semibold text-green-700 mb-2 text-center">
          Riwayat Hasil Klasifikasi (Sementara)
        </h3>
        <p className="text-gray-500 text-sm text-center mb-5">
          Data akan hilang setelah halaman direfresh.
        </p>

        {history.length === 0 ? (
          <p className="text-center text-gray-400 italic text-sm py-4">
            Belum ada hasil klasifikasi.
          </p>
        ) : (
          <div className="space-y-3">
            {history.map((item, idx) => (
              <HistoryItem key={item.id} item={item} index={idx} onDelete={onDelete} />
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
