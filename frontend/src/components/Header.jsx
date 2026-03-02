import { motion as Motion } from 'framer-motion'
import PropTypes from 'prop-types'

const Header = ({ health, mode, onModeChange }) => {
  const isHealthy = health === 'healthy'

  return (
    <header>
      <div className="logo-section">
        <Motion.h1
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          AL Financial <span style={{ color: '#111827' }}>OCR</span>
        </Motion.h1>
      </div>
      <div className="header-controls">
        <div className="mode-switch">
          <button
            type="button"
            className={`mode-btn ${mode === 'ocr' ? 'active' : ''}`}
            onClick={() => onModeChange('ocr')}
          >
            OCR
          </button>
          <button
            type="button"
            className={`mode-btn ${mode === 'dataset' ? 'active' : ''}`}
            onClick={() => onModeChange('dataset')}
          >
            Dataset
          </button>
        </div>
        <div className="status-badge">
          <div
            className={`status-dot ${isHealthy ? '' : 'offline'}`}
            style={{ backgroundColor: isHealthy ? '#22c55e' : '#ef4444' }}
          />
          <span>System {health}</span>
        </div>
      </div>
    </header>
  )
}

Header.propTypes = {
  health: PropTypes.string,
  mode: PropTypes.oneOf(['ocr', 'dataset']),
  onModeChange: PropTypes.func.isRequired,
}

Header.defaultProps = {
  health: 'unknown',
  mode: 'ocr',
}

export default Header
