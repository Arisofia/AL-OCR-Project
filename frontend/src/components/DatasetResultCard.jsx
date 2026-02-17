import { FileText, Loader2, CheckCircle2, AlertTriangle } from 'lucide-react'
import { motion as Motion, AnimatePresence } from 'framer-motion'

const DatasetResultCard = ({ uploads, uploading }) => {
  const rows = uploads || []

  return (
    <section className="result-section">
      <div className="result-card">
        <div className="result-header">
          <h2>Dataset Uploads</h2>
          <div className="stats">
            <span>{rows.length} items</span>
            {uploading ? <span>in progress</span> : <span>idle</span>}
          </div>
        </div>

        <AnimatePresence mode="wait">
          {uploading && rows.length === 0 ? (
            <Motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="empty-state"
            >
              <Loader2 className="animate-spin" size={48} color="#e2e8f0" />
              <p>Uploading files...</p>
            </Motion.div>
          ) : rows.length ? (
            <Motion.div
              key="rows"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="result-content"
              style={{ display: 'flex', flexDirection: 'column', flex: 1, gap: '0.75rem' }}
            >
              {rows.map((row) => (
                <div
                  key={row.id}
                  className="upload-row"
                >
                  <div className="upload-row__left">
                    {row.status === 'success' ? (
                      <CheckCircle2 size={16} color="#16a34a" />
                    ) : row.status === 'error' ? (
                      <AlertTriangle size={16} color="#dc2626" />
                    ) : (
                      <Loader2 size={16} className="animate-spin" color="#64748b" />
                    )}
                    <div className="upload-row__meta">
                      <div className="upload-row__name">{row.fileName}</div>
                      <div className="upload-row__sub">
                        {row.s3_key ? <code>{row.s3_key}</code> : row.detail || 'pending'}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </Motion.div>
          ) : (
            <Motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="empty-state"
            >
              <FileText size={48} color="#e2e8f0" />
              <p>No dataset uploads yet</p>
            </Motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  )
}

export default DatasetResultCard

