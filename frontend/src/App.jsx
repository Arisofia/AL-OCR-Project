import { useState, useEffect } from 'react'
import api from './api'
import Header from './components/Header'
import UploadCard from './components/UploadCard'
import ResultCard from './components/ResultCard'
import DatasetUploadCard from './components/DatasetUploadCard'
import DatasetResultCard from './components/DatasetResultCard'
import './App.css'

function App() {
  const [mode, setMode] = useState('ocr')
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState(null)
  const [health, setHealth] = useState('checking')
  const [ocrDocType, setOcrDocType] = useState('generic')
  const [useReconstruction, setUseReconstruction] = useState(false)

  const [datasetKey, setDatasetKey] = useState('')
  const [datasetFiles, setDatasetFiles] = useState([])
  const [dataset, setDataset] = useState('occlusion_cards')
  const [split, setSplit] = useState('inbox')
  const [docType, setDocType] = useState('bank_card')
  const [occlusionType, setOcclusionType] = useState('finger')
  const [notes, setNotes] = useState('')
  const [datasetUploads, setDatasetUploads] = useState([])
  const [datasetUploading, setDatasetUploading] = useState(false)

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await api.get('/health')
        const status = String(response?.data?.status || '').toLowerCase()
        if (status === 'healthy' || status === 'ok' || status === 'degraded') {
          setHealth('healthy')
        } else {
          setHealth('unhealthy')
        }
      } catch (error) {
        // If it's a 403 (missing key) but the endpoint exists, it's technically 'online'
        if (error.response?.status === 403) {
          setHealth('healthy')
        } else {
          setHealth('offline')
        }
      }
    }
    checkHealth()
    const timer = setInterval(checkHealth, 5000)
    return () => clearInterval(timer)
  }, [])

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    const isImage = Boolean(selectedFile?.type?.startsWith('image/'))
    const isPdf = selectedFile?.type === 'application/pdf'

    if (selectedFile && (isImage || isPdf)) {
      setFile(selectedFile)
      if (preview) URL.revokeObjectURL(preview)
      setPreview(isImage ? URL.createObjectURL(selectedFile) : null)
      setResult(null)
    } else if (selectedFile) {
      alert('Please select a valid image or PDF file.')
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    const createPayload = () => {
      const formData = new FormData()
      formData.append('file', file)
      return formData
    }

    const requestOcr = (docTypeValue) =>
      api.post('/ocr', createPayload(), {
        params: {
          doc_type: docTypeValue,
          reconstruct: useReconstruction
        },
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

    try {
      const response = await requestOcr(ocrDocType)
      setResult(response.data)
    } catch (error) {
      let currentError = error
      const shouldRetryWithGeneric =
        ocrDocType === 'bank_card' &&
        [502, 504].includes(currentError?.response?.status)

      if (shouldRetryWithGeneric) {
        try {
          const fallbackResponse = await requestOcr('generic')
          setResult(fallbackResponse.data)
          alert('Bank Card mode timed out. Generic OCR fallback was used.')
          return
        } catch (fallbackError) {
          currentError = fallbackError
        }
      }

      console.error('OCR Error:', currentError)
      const responseData = currentError.response?.data
      const errorDetail =
        typeof responseData === 'string'
          ? responseData
          : responseData?.detail
      const errorMsg =
        errorDetail || currentError.message || 'Error processing document'
      alert(`Status ${currentError.response?.status || 'Unknown'}: ${errorMsg}`)
    } finally {
      setLoading(false)
    }
  }

  const handleDatasetFilesChange = (e) => {
    const files = Array.from(e.target.files || [])
    setDatasetFiles(files)
    setDatasetUploads([])
  }

  const handleDatasetUpload = async () => {
    if (!datasetKey || datasetFiles.length === 0) return

    setDatasetUploading(true)

    const baseId = Date.now()
    const initial = datasetFiles.map((f, idx) => ({
      id: `${baseId}-${idx}`,
      fileName: f.name,
      status: 'pending'
    }))
    setDatasetUploads(initial)

    for (let idx = 0; idx < datasetFiles.length; idx++) {
      const f = datasetFiles[idx]
      const rowId = `${baseId}-${idx}`
      setDatasetUploads((prev) =>
        prev.map((r) => (r.id === rowId ? { ...r, status: 'uploading' } : r))
      )

      const formData = new FormData()
      formData.append('file', f)
      formData.append('dataset', dataset)
      formData.append('split', split)
      formData.append('doc_type', docType)
      formData.append('occlusion_type', occlusionType)
      formData.append('use_reconstruction', String(useReconstruction))
      formData.append('notes', notes)

      try {
        const response = await api.post('/datasets/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
            'X-DATASET-KEY': datasetKey
          }
        })
        setDatasetUploads((prev) =>
          prev.map((r) =>
            r.id === rowId
              ? { ...r, status: 'success', ...response.data }
              : r
          )
        )
      } catch (error) {
        const errorMsg =
          error.response?.data?.detail || 'Error uploading dataset image'
        setDatasetUploads((prev) =>
          prev.map((r) =>
            r.id === rowId ? { ...r, status: 'error', detail: errorMsg } : r
          )
        )
      }
    }

    setDatasetUploading(false)
  }

  return (
    <div className="app-container">
      <Header
        health={health}
        mode={mode}
        onModeChange={setMode}
      />
      {mode === 'ocr' ? (
        <main className="main-grid">
          <UploadCard
            file={file}
            preview={preview}
            loading={loading}
            health={health}
            docType={ocrDocType}
            useReconstruction={useReconstruction}
            onReconstructionChange={setUseReconstruction}
            onDocTypeChange={setOcrDocType}
            onFileChange={handleFileChange}
            onUpload={handleUpload}
          />
          <ResultCard
            result={result}
            loading={loading}
          />
        </main>
      ) : (
        <main className="main-grid">
          <DatasetUploadCard
            files={datasetFiles}
            uploading={datasetUploading}
            health={health}
            datasetKey={datasetKey}
            dataset={dataset}
            split={split}
            docType={docType}
            occlusionType={occlusionType}
            useReconstruction={useReconstruction}
            onReconstructionChange={setUseReconstruction}
            notes={notes}
            onDatasetKeyChange={setDatasetKey}
            onDatasetChange={setDataset}
            onSplitChange={setSplit}
            onDocTypeChange={setDocType}
            onOcclusionTypeChange={setOcclusionType}
            onNotesChange={setNotes}
            onFilesChange={handleDatasetFilesChange}
            onUpload={handleDatasetUpload}
          />
          <DatasetResultCard
            uploads={datasetUploads}
            uploading={datasetUploading}
          />
        </main>
      )}
    </div>
  )
}

export default App
