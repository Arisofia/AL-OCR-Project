import { Upload, Loader2 } from 'lucide-react'

const DatasetUploadCard = ({
  files,
  uploading,
  health,
  datasetKey,
  dataset,
  split,
  docType,
  occlusionType,
  notes,
  onDatasetKeyChange,
  onDatasetChange,
  onSplitChange,
  onDocTypeChange,
  onOcclusionTypeChange,
  onNotesChange,
  onFilesChange,
  onUpload
}) => {
  const fileCount = files?.length || 0

  return (
    <section className="upload-section">
      <div className="upload-card">
        <div className="card-title">
          <h2>Dataset Upload</h2>
          <p className="hint">
            Protected endpoint. Requires <code>X-DATASET-KEY</code>. Do not upload real cardholder data.
          </p>
        </div>

        <div className="form-stack">
          <label className="field">
            <span>Dataset key</span>
            <input
              className="text-input"
              type="password"
              value={datasetKey}
              onChange={(e) => onDatasetKeyChange(e.target.value)}
              placeholder="Paste X-DATASET-KEY"
              autoComplete="off"
            />
          </label>

          <div className="row">
            <label className="field">
              <span>Dataset</span>
              <input
                className="text-input"
                value={dataset}
                onChange={(e) => onDatasetChange(e.target.value)}
              />
            </label>
            <label className="field">
              <span>Split</span>
              <select
                className="text-input"
                value={split}
                onChange={(e) => onSplitChange(e.target.value)}
              >
                <option value="inbox">inbox</option>
                <option value="holdout">holdout</option>
              </select>
            </label>
          </div>

          <div className="row">
            <label className="field">
              <span>Doc type</span>
              <select
                className="text-input"
                value={docType}
                onChange={(e) => onDocTypeChange(e.target.value)}
              >
                <option value="bank_card">bank_card</option>
                <option value="invoice">invoice</option>
                <option value="receipt">receipt</option>
                <option value="id_document">id_document</option>
                <option value="generic">generic</option>
              </select>
            </label>
            <label className="field">
              <span>Occlusion</span>
              <select
                className="text-input"
                value={occlusionType}
                onChange={(e) => onOcclusionTypeChange(e.target.value)}
              >
                <option value="finger">finger</option>
                <option value="shadow">shadow</option>
                <option value="glare">glare</option>
                <option value="blur">blur</option>
                <option value="unknown">unknown</option>
              </select>
            </label>
          </div>

          <label className="field">
            <span>Notes</span>
            <textarea
              className="text-input textarea"
              value={notes}
              onChange={(e) => onNotesChange(e.target.value)}
              placeholder="Optional notes (camera, angle, partial coverage, etc.)"
              rows={3}
            />
          </label>
        </div>

        <div className="dropzone">
          <input type="file" onChange={onFilesChange} accept="image/*" multiple />
          <div className="dropzone-content">
            <div className="icon-circle">
              <Upload size={32} />
            </div>
            {fileCount ? (
              <p><strong>{fileCount} file(s) selected</strong></p>
            ) : (
              <p><strong>Click to select images</strong> (multiple supported)</p>
            )}
            <span>JPEG/PNG recommended</span>
          </div>
        </div>

        <button
          className="btn-primary"
          onClick={onUpload}
          disabled={
            uploading ||
            health !== 'healthy' ||
            !datasetKey ||
            fileCount === 0
          }
        >
          {uploading ? (
            <>
              <Loader2 className="loading-pulse" size={20} />
              Uploading...
            </>
          ) : (
            <>Upload To Dataset</>
          )}
        </button>

        {health !== 'healthy' && (
          <p className="hint" style={{ marginTop: '0.75rem' }}>
            Backend health is <strong>{health}</strong>. Uploads disabled.
          </p>
        )}
      </div>
    </section>
  )
}

export default DatasetUploadCard
