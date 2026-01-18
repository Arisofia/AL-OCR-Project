import { Upload, Loader2, Zap } from 'lucide-react'
import { motion as Motion } from 'framer-motion'

const UploadCard = ({ file, preview, loading, health, onFileChange, onUpload }) => {
  return (
    <section className="upload-section">
      <div className="upload-card">
        <div className="dropzone">
          <input type="file" onChange={onFileChange} accept="image/*" />
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
          onClick={onUpload}
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
  )
}

export default UploadCard
