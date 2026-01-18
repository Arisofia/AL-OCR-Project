import { useState, useEffect } from 'react'
import api from './api'
import Header from './components/Header'
import UploadCard from './components/UploadCard'
import ResultCard from './components/ResultCard'
import './App.css'

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
      <Header health={health} />
      <main className="main-grid">
        <UploadCard
          file={file}
          preview={preview}
          loading={loading}
          health={health}
          onFileChange={handleFileChange}
          onUpload={handleUpload}
        />
        <ResultCard
          result={result}
          loading={loading}
        />
      </main>
    </div>
  )
}

export default App
