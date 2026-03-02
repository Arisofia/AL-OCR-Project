import { FileText, Loader2 } from 'lucide-react'
import { motion as Motion, AnimatePresence } from 'framer-motion'
import PropTypes from 'prop-types'

const ResultCard = ({ result, loading }) => {
  const cardAnalysis = result?.card_analysis

  const renderContent = () => {
    if (loading) {
      return (
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
      )
    }

    if (result) {
      return (
        <Motion.div
          key="result"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="result-content"
          style={{ display: 'flex', flexDirection: 'column', flex: 1 }}
        >
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.75rem' }}>
            {result.document_type && (
              <span style={{
                fontSize: '0.75rem',
                padding: '2px 8px',
                background: '#f1f5f9',
                border: '1px solid #e2e8f0',
                borderRadius: '9999px',
                color: '#334155'
              }}>
                Doc: <strong>{result.document_type}</strong>
              </span>
            )}
            {cardAnalysis?.detected && (
              <span style={{
                fontSize: '0.75rem',
                padding: '2px 8px',
                background: cardAnalysis.requires_manual_review ? '#fff7ed' : '#ecfeff',
                border: '1px solid #e2e8f0',
                borderRadius: '9999px',
                color: '#334155'
              }}>
                Card candidates: <strong>{cardAnalysis.candidate_count}</strong> · Luhn valid:{' '}
                <strong>{cardAnalysis.luhn_valid_count}</strong>
                {cardAnalysis.requires_manual_review ? ' · Manual review' : ''}
              </span>
            )}
          </div>
          <div className="iteration-pills" style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
            {result.iterations?.map((it) => (
              <span key={`it-${it.iteration}`} style={{
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
          {cardAnalysis?.candidates?.length > 0 && (
            <div style={{ marginBottom: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              {cardAnalysis.candidates.map((row) => (
                <div key={row.masked} style={{ fontSize: '0.8rem', color: '#475569' }}>
                  <code style={{
                    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                    fontSize: '0.75rem',
                    background: '#f8fafc',
                    border: '1px solid #e2e8f0',
                    padding: '2px 6px',
                    borderRadius: '6px',
                    marginRight: '0.5rem',
                    color: '#0f172a'
                  }}>
                    {row.masked}
                  </code>
                  <span>
                    brand={row.brand_guess || 'unknown'} · len={row.length} · luhn={String(row.luhn_valid)}
                  </span>
                </div>
              ))}
            </div>
          )}
          <pre className="ocr-text">{result.text}</pre>
        </Motion.div>
      )
    }

    return (
      <Motion.div
        key="empty"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="empty-state"
      >
        <FileText size={48} color="#e2e8f0" />
        <p>Upload a document to see extracted text</p>
      </Motion.div>
    )
  }

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
          {renderContent()}
        </AnimatePresence>
      </div>
    </section>
  )
}

ResultCard.propTypes = {
  result: PropTypes.shape({
    card_analysis: PropTypes.shape({
      detected: PropTypes.bool,
      candidate_count: PropTypes.number,
      luhn_valid_count: PropTypes.number,
      requires_manual_review: PropTypes.bool,
      candidates: PropTypes.arrayOf(
        PropTypes.shape({
          masked: PropTypes.string,
          brand_guess: PropTypes.string,
          length: PropTypes.number,
          luhn_valid: PropTypes.bool,
        })
      ),
    }),
    iterations: PropTypes.arrayOf(
      PropTypes.shape({
        iteration: PropTypes.number,
        text_length: PropTypes.number,
      })
    ),
    processing_time: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    document_type: PropTypes.string,
    text: PropTypes.string,
  }),
  loading: PropTypes.bool,
}

ResultCard.defaultProps = {
  result: null,
  loading: false,
}

export default ResultCard
