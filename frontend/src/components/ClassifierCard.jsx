import { useEffect, useRef, useState } from 'react'
import { Camera, Upload, RotateCcw, Sparkles, AlertTriangle, CheckCircle2, X } from 'lucide-react'

const API_BASE_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

export default function ClassifierCard({ onNewResult }) {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isCameraOpen, setIsCameraOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [result, setResult] = useState(null)

  const fileInputRef = useRef(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)

  // Cleanup object URL & camera stream on unmount
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      stopCamera()
    }
  }, [])

  const updateSelectedFile = (file) => {
    if (!file) return
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setErrorMsg(null)
  }

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      updateSelectedFile(file)
      stopCamera()
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) {
      updateSelectedFile(file)
      stopCamera()
    } else {
      setErrorMsg('Mohon unggah file gambar yang valid (JPG, PNG, WEBP).')
    }
  }

  const startCamera = async () => {
    setErrorMsg(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      })
      streamRef.current = stream
      setIsCameraOpen(true)
      // Wait for video element to mount
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      })
    } catch (err) {
      console.error('startCamera error:', err)
      setErrorMsg('Tidak dapat mengakses kamera. Pastikan izin diberikan dan perangkat memiliki kamera.')
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsCameraOpen(false)
  }

  const capturePhoto = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    const w = video.videoWidth || 640
    const h = video.videoHeight || 480
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, w, h)

    canvas.toBlob(
      (blob) => {
        if (!blob) return
        const file = new File([blob], `capture_${Date.now()}.jpg`, {
          type: blob.type || 'image/jpeg',
        })
        updateSelectedFile(file)
        stopCamera()
      },
      'image/jpeg',
      0.95
    )
  }

  const handleClassify = async () => {
    if (!selectedFile) {
      setErrorMsg('Mohon pilih atau ambil foto mangga terlebih dahulu.')
      return
    }

    setIsLoading(true)
    setErrorMsg(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch(`${API_BASE_URL}/classify`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok || data.error) {
        setErrorMsg(data.error || 'Terjadi kesalahan pada server.')
        return
      }

      if (data.is_mango === false || data.label === 'Bukan buah mangga') {
        setResult({
          isMango: false,
          label: data.label || 'Bukan buah mangga',
          confidence: null,
          predictions: [],
        })
        return
      }

      const acc = (data.confidence * 100).toFixed(2)
      const parsedResult = {
        isMango: true,
        label: data.label,
        confidence: acc,
        predictions: data.predictions || [],
      }

      setResult(parsedResult)

      onNewResult({
        id: `h_${Date.now()}`,
        waktu: new Date().toLocaleString('id-ID'),
        label: parsedResult.label,
        akurasi: `${acc}%`,
        predictions: parsedResult.predictions,
        imageSrc: previewUrl,
      })
    } catch (err) {
      console.error('Classify error:', err)
      setErrorMsg('Gagal menghubungi server klasifikasi. Pastikan backend Flask berjalan.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    stopCamera()
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setSelectedFile(null)
    setPreviewUrl(null)
    setResult(null)
    setErrorMsg(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <main id="klasifikasi" className="container mx-auto py-12 px-4">
      <div className="bg-white rounded-2xl shadow-md p-6 md:p-10 flex flex-col items-center max-w-3xl mx-auto">
        <h3 className="text-2xl font-semibold text-green-700 mb-6">Upload Gambar Mangga</h3>

        {/* Drop Zone */}
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`w-full border-2 border-dashed rounded-xl h-48 flex flex-col justify-center items-center transition-colors p-4 ${
            isDragging
              ? 'border-green-600 bg-green-100'
              : 'border-green-400 bg-green-50 hover:bg-green-100/70'
          } ${isCameraOpen ? 'opacity-60' : ''}`}
        >
          <p className="text-gray-600 text-sm md:text-base text-center">
            Seret & lepas gambar di sini, atau pilih melalui tombol di bawah
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <div className="mt-4 flex flex-wrap justify-center gap-3">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition cursor-pointer text-sm font-medium"
            >
              <Upload className="w-4 h-4" />
              Pilih Gambar
            </button>
            <button
              type="button"
              onClick={startCamera}
              className="inline-flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition cursor-pointer text-sm font-medium"
            >
              <Camera className="w-4 h-4" />
              Gunakan Kamera
            </button>
          </div>
        </div>

        {/* Camera Stream */}
        {isCameraOpen && (
          <div className="mt-5 flex flex-col items-center">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-72 h-52 bg-black rounded-lg object-cover shadow"
            />
            <div className="mt-3 flex gap-3">
              <button
                type="button"
                onClick={capturePhoto}
                className="bg-yellow-500 text-white px-4 py-2 rounded-lg hover:bg-yellow-600 transition cursor-pointer text-sm font-medium"
              >
                Ambil Foto
              </button>
              <button
                type="button"
                onClick={stopCamera}
                className="inline-flex items-center gap-1 bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition cursor-pointer text-sm font-medium"
              >
                <X className="w-4 h-4" />
                Batal
              </button>
            </div>
          </div>
        )}

        <canvas ref={canvasRef} className="hidden" />

        {/* Image Preview */}
        {previewUrl && (
          <div className="mt-6 flex flex-col items-center">
            <img
              src={previewUrl}
              alt="Preview Mangga"
              className="w-64 h-64 object-cover rounded-xl shadow-md border border-gray-200"
            />
            {selectedFile && (
              <p className="text-xs text-gray-500 mt-2">{selectedFile.name}</p>
            )}
          </div>
        )}

        {/* Error Alert */}
        {errorMsg && (
          <div className="mt-5 w-full bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Classify Button */}
        <button
          type="button"
          onClick={handleClassify}
          disabled={isLoading || !selectedFile}
          className="mt-6 inline-flex items-center gap-2 bg-yellow-500 text-white px-6 py-2.5 rounded-lg hover:bg-yellow-600 disabled:opacity-50 disabled:cursor-not-allowed transition cursor-pointer font-medium shadow-sm"
        >
          <Sparkles className="w-4 h-4" />
          {isLoading ? 'Memproses...' : 'Klasifikasikan'}
        </button>

        {/* Result Box */}
        {result && (
          <div className="mt-6 w-full bg-gray-50 border border-gray-200 rounded-xl p-5 text-center">
            {!result.isMango ? (
              <div className="flex flex-col items-center text-amber-700">
                <AlertTriangle className="w-8 h-8 mb-2" />
                <p className="text-lg font-semibold">{result.label}</p>
                <p className="text-sm text-gray-600 mt-1">
                  Confidence model berada di bawah ambang batas (90%). Mohon unggah foto buah mangga yang jelas.
                </p>
              </div>
            ) : (
              <div>
                <div className="flex items-center justify-center gap-2 text-green-700">
                  <CheckCircle2 className="w-5 h-5" />
                  <h4 className="text-base font-medium">Hasil Klasifikasi</h4>
                </div>
                <p className="text-2xl font-bold text-green-700 mt-1">{result.label}</p>
                <p className="text-sm text-gray-600 mt-1">Akurasi: {result.confidence}%</p>

                {result.predictions.length > 0 && (
                  <div className="mt-4 text-left space-y-2 border-t border-gray-200 pt-4">
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                      Distribusi Probabilitas
                    </p>
                    {result.predictions.map((pred) => {
                      const pct = (pred.p * 100).toFixed(2)
                      return (
                        <div key={pred.label} className="text-xs">
                          <div className="flex justify-between mb-1">
                            <span className="font-medium text-gray-700">{pred.label}</span>
                            <span className="text-gray-600">{pct}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-green-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${Math.min(100, Number(pct))}%` }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Reset Button */}
        {(selectedFile || result || errorMsg) && (
          <button
            type="button"
            onClick={handleReset}
            className="mt-5 inline-flex items-center gap-1.5 text-sm text-gray-600 hover:text-green-700 underline cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
        )}
      </div>
    </main>
  )
}
