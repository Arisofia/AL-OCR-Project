import { motion as Motion } from 'framer-motion'

const Header = ({ health }) => {
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
      <div className="status-badge">
        <div className={`status-dot ${health !== 'healthy' ? 'offline' : ''}`}
             style={{ backgroundColor: health === 'healthy' ? '#22c55e' : '#ef4444' }} />
        <span>System {health}</span>
      </div>
    </header>
  )
}

export default Header
