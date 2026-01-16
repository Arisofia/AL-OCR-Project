import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  Upload,
  FileText,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Image as ImageIcon,
  Zap
} from 'lucide-react'
import { motion as Motion, AnimatePresence } from 'framer-motion'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
const IS_DEV = import.meta.env.MODE === 'development'
// Prefer an explicit env var for the API key. In development only, fall back to a
// known dev key to make local runs easier. In non-dev environments, fail fast
// to avoid silent misconfiguration.
const API_KEY = import.meta.env.VITE_API_KEY || (IS_DEV ? 'default_secret_key' : undefined)

if (!API_KEY && !IS_DEV) {
  console.error('[Config Error] VITE_API_KEY is not set. Define VITE_API_KEY in your environment.')
  throw new Error('Missing VITE_API_KEY environment variable')
}

// Initialize axios instance for enterprise-grade consistency
const apiHeaders = {
  Accept: 'application/json',
}
if (API_KEY) {
  apiHeaders['X-API-KEY'] = API_KEY
}

const api = axios.create({
  baseURL: API_BASE,
  headers: apiHeaders,
})

function App() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState(null)
  const [health, setHealth] = useState('checking')

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await api.get('/health')
        if (response.data.status === 'healthy') {
          setHealth('healthy')
        } else {
          setHealth('unhealthy')
        }
      } catch {
        setHealth('offline')
      }
    }
    checkHealth()
  }, [])

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setFile(selectedFile)
      if (preview) URL.revokeObjectURL(preview)
      setPreview(URL.createObjectURL(selectedFile))
      setResult(null)
    } else if (selectedFile) {
      alert('Please select a valid image file.')
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await api.post('/ocr', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      setResult(response.data)
    } catch (error) {
      console.error('OCR Error:', error)
      const errorMsg = error.response?.data?.detail || 'Error processing document'
      alert(`Status ${error.response?.status || 'Unknown'}: ${errorMsg}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header>
        <div className="logo-section">
          <Motion.h1
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            AL Financial <span style={{ color: '#111827' }}>OCR</span>
          </Motion.h1>
        </div>
        <div className="status-badge">
          <div className={`status-dot ${health !== 'healthy' ? 'offline' : ''}`}
               style={{ backgroundColor: health === 'healthy' ? '#22c55e' : '#ef4444' }} />
          <span>System {health}</span>
        </div>
      </header>

      <main className="main-grid">
        <section className="upload-section">
          <div className="upload-card">
            <div className="dropzone">
              <input type="file" onChange={handleFileChange} accept="image/*" />
              <div className="dropzone-content">
                {preview ? (
                  <Motion.img
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    src={preview}
                    className="preview-image"
                    alt="Preview"
                  />
                ) : (
                  <>
                    <div className="icon-circle">
                      <Upload size={32} />
                    </div>
                    <p><strong>Click to upload</strong> or drag and drop</p>
                    <span>Financial documents, receipts, or IDs</span>
                  </>
                )}
              </div>
            </div>

            <button
              className="btn-primary"
              onClick={handleUpload}
              disabled={loading || !file || health !== 'healthy'}
            >
              {loading ? (
                <>
                  <Loader2 className="loading-pulse" size={20} />
                  Analyzing Document...
                </>
              ) : (
                <>
                  <Zap size={20} fill="currentColor" />
                  Extract Data
                </>
              )}
            </button>
          </div>
        </section>

        <section className="result-section">
          <div className="result-card">
            <div className="result-header">
              <h2>Extracted Information</h2>
              {result && (
                <div className="stats">
                  <span>{result.iterations?.length} iterations</span>
                  <span>{result.processing_time}s</span>
                </div>
              )}
            </div>

            <AnimatePresence mode="wait">
              {loading ? (
                <Motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="empty-state"
                >
                  <Loader2 className="animate-spin" size={48} color="#e2e8f0" />
                  <p>Our AI is reading your document...</p>
                </Motion.div>
              ) : result ? (
                <Motion.div
                  key="result"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="result-content"
                  style={{ display: 'flex', flexDirection: 'column', flex: 1 }}
                >
                  <div className="iteration-pills" style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                    {result.iterations?.map((it) => (
                      <span key={it.iteration} style={{
                        fontSize: '0.7rem',
                        padding: '2px 8px',
                        background: '#e2e8f0',
                        borderRadius: '4px',
                        color: '#475569'
                      }}>
                        Iteration {it.iteration}: {it.text_length} chars
                      </span>
                    ))}
                  </div>
                  <pre className="ocr-text">{result.text}</pre>
                </Motion.div>
              ) : (
                <Motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="empty-state"
                >
                  <FileText size={48} color="#e2e8f0" />
                  <p>Upload a document to see extracted text</p>
                </Motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
