import { FileText, Loader2 } from 'lucide-react'
import { motion as Motion, AnimatePresence } from 'framer-motion'

const ResultCard = ({ result, loading }) => {
  return (
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
  )
}

export default ResultCard
