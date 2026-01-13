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
import { motion, AnimatePresence } from 'framer-motion'
import './App.css'

const API_BASE = 'http://localhost:8000'
const API_KEY = 'default_secret_key'

function App() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState(null)
  const [health, setHealth] = useState('checking')

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get(`${API_BASE}/health`)
        if (response.data.status === 'healthy') {
          setHealth('healthy')
        } else {
          setHealth('unhealthy')
        }
      } catch (error) {
        setHealth('offline')
      }
    }
    checkHealth()
  }, [])

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setFile(selectedFile)
      setPreview(URL.createObjectURL(selectedFile))
      setResult(null)
    } else {
      alert('Please select a valid image file.')
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post(`${API_BASE}/ocr`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'X-API-KEY': API_KEY
        }
      })
      setResult(response.data)
    } catch (error) {
      console.error('OCR Error:', error)
      alert(error.response?.data?.detail || 'Error processing document')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header>
        <div className="logo-section">
          <motion.h1 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            AL Financial <span style={{ color: '#111827' }}>OCR</span>
          </motion.h1>
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
                  <motion.img 
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
                  <span><Zap size={12} style={{ display: 'inline' }} /> {result.strategy_used}</span>
                  <span>{result.processing_time}s</span>
                </div>
              )}
            </div>

            <AnimatePresence mode="wait">
              {loading ? (
                <motion.div 
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="empty-state"
                >
                  <Loader2 className="animate-spin" size={48} color="#e2e8f0" />
                  <p>Our AI is reading your document...</p>
                </motion.div>
              ) : result ? (
                <motion.div 
                  key="result"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="result-content"
                  style={{ display: 'flex', flexDirection: 'column', flex: 1 }}
                >
                  <pre className="ocr-text">{result.text}</pre>
                </motion.div>
              ) : (
                <motion.div 
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="empty-state"
                >
                  <FileText size={48} color="#e2e8f0" />
                  <p>Upload a document to see extracted text</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
