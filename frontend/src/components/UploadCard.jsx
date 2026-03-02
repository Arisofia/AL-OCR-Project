import { Upload, Loader2, Zap, FileText } from 'lucide-react'
import { motion } from 'framer-motion'
import PropTypes from 'prop-types'

const MotionImg = motion.img

const UploadCard = ({
  file,
  preview,
  loading,
  health,
  docType,
  onDocTypeChange,
  onFileChange,
  onUpload
}) => {
  const renderDropzoneContent = () => {
    if (preview) {
      return (
        <MotionImg
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          src={preview}
          className="preview-image"
          alt="Preview"
        />
      )
    }

    if (file) {
      return (
        <>
          <div className="icon-circle">
            <FileText size={32} />
          </div>
          <p><strong>{file.name}</strong></p>
          <span>Ready to extract</span>
        </>
      )
    }

    return (
      <>
        <div className="icon-circle">
          <Upload size={32} />
        </div>
        <p><strong>Click to upload</strong> or drag and drop</p>
        <span>Financial documents, receipts, or IDs (images or PDF)</span>
      </>
    )
  }

  return (
    <section className="upload-section">
      <div className="upload-card">
        <div className="dropzone">
          <input type="file" onChange={onFileChange} accept="image/*,application/pdf" />
          <div className="dropzone-content">
            {renderDropzoneContent()}
          </div>
        </div>

        <div className="form-stack" style={{ marginTop: '1rem' }}>
          <div className="field">
            <span>Document Type</span>
            <select
              className="text-input"
              value={docType}
              onChange={(e) => onDocTypeChange(e.target.value)}
            >
              <option value="generic">Auto / Generic</option>
              <option value="bank_card">Bank Card</option>
              <option value="invoice">Invoice</option>
              <option value="receipt">Receipt</option>
              <option value="id_document">ID Document</option>
            </select>
          </div>
          <p className="hint" style={{ margin: 0 }}>
            For cards, select <code>Bank Card</code> to enable card-optimized OCR (padding + digit rescue).
          </p>
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

UploadCard.propTypes = {
  file: PropTypes.instanceOf(File),
  preview: PropTypes.string,
  loading: PropTypes.bool,
  health: PropTypes.string,
  docType: PropTypes.string,
  onDocTypeChange: PropTypes.func.isRequired,
  onFileChange: PropTypes.func.isRequired,
  onUpload: PropTypes.func.isRequired,
}

UploadCard.defaultProps = {
  file: null,
  preview: null,
  loading: false,
  health: 'unknown',
  docType: 'generic',
}

export default UploadCard
