import { useState } from 'react';
import { Database, Upload } from 'lucide-react';
import { Panel } from '../components/Panel';
import { SourceBadge } from '../components/SourceBadge';
import { api, errorMessage, fmtBytes, type ImportSummary } from '../lib/api';
import { useApi } from '../lib/useApi';

export default function Datasets() {
  const datasets = useApi(() => api.datasets());
  const [file, setFile] = useState<File | null>(null);
  const [overwrite, setOverwrite] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState<ImportSummary | null>(null);

  async function upload() {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setUploaded(await api.importFile(file, overwrite));
      datasets.reload();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page-stack">
      <div className="page-heading"><div><h1>Datasets</h1><p>Local imports the API can forecast from, plus the documented external sources. Raw motorsport data never enters git.</p></div></div>
      <div className="two-col">
        <Panel title="Local imports (data/imports)" icon={<Database size={17} />} action={datasets.error ? <SourceBadge kind="offline" /> : <SourceBadge kind="local" />}>
          {datasets.error && <div className="error-box">{datasets.error}</div>}
          {datasets.data && datasets.data.imports.length === 0 && <div className="empty-inline">No CSV or Parquet files yet.</div>}
          {datasets.data && datasets.data.imports.length > 0 && (
            <div className="wide-table imports-table">
              <div className="wide-head"><span>File</span><span>Size</span><span>Modified (UTC)</span></div>
              {datasets.data.imports.map(i => <div className="wide-row" key={i.name}><span><strong>{i.name}</strong></span><span>{fmtBytes(i.bytes)}</span><span>{i.modified_utc.replace('T', ' ').slice(0, 19)}</span></div>)}
            </div>
          )}
          {datasets.data && datasets.data.processed_files.filter(f => !f.startsWith('.')).length > 0 && (
            <div className="detail-list compact"><div><span>Processed tables</span><strong>{datasets.data.processed_files.filter(f => !f.startsWith('.')).join(', ')}</strong></div></div>
          )}
        </Panel>
        <Panel title="Import a lap table" icon={<Upload size={17} />}>
          <div className="form-grid single">
            <label className="field"><span>CSV or Parquet (max 200 MB)</span><input type="file" accept=".csv,.parquet" onChange={e => setFile(e.target.files?.[0] ?? null)} /></label>
            <label className="check-inline"><input type="checkbox" checked={overwrite} onChange={e => setOverwrite(e.target.checked)} /> Replace an existing file with the same name</label>
            <button className="primary-btn" onClick={upload} disabled={!file || busy}><Upload size={15} />{busy ? 'Uploading…' : 'Import into data/imports'}</button>
          </div>
          {error && <div className="error-box">{error}</div>}
          {uploaded && (
            <div className="detail-list compact">
              <div><span>Stored as</span><strong>{uploaded.stored_as ?? uploaded.file}</strong></div>
              <div><span>Rows</span><strong>{uploaded.rows.toLocaleString()}</strong></div>
              <div><span>Required columns</span><strong className={uploaded.missing_required_columns.length ? 'bad' : 'good'}>{uploaded.missing_required_columns.length ? `missing ${uploaded.missing_required_columns.join(', ')}` : 'all present'}</strong></div>
              {uploaded.latest_session && <div><span>Latest session</span><strong>{uploaded.latest_session.season} · {uploaded.latest_session.event}</strong></div>}
            </div>
          )}
          <p className="prose small">Schema: see <code>docs/FEATURE_CONTRACT.md</code>. Required columns: season, event, session, driver, lap_number, lap_time_s.</p>
        </Panel>
      </div>
      <Panel title="Source registry (dataset_manifest.json)" icon={<Database size={17} />}>
        {datasets.data && (
          <div className="wide-table sources-table">
            <div className="wide-head"><span>Source</span><span>Status</span><span>Role</span><span>Coverage</span><span>Local policy</span></div>
            {datasets.data.sources.map(s => (
              <div className="wide-row" key={s.name}>
                <span><strong><a href={s.url} target="_blank" rel="noreferrer">{s.name}</a></strong></span>
                <span>{s.status === 'used' ? <SourceBadge kind="real" text="Used in results" /> : <SourceBadge kind="fixture" text="Planned, not read" />}</span>
                <span>{s.role}</span>
                <span>{s.coverage_note}</span>
                <span>{s.local_policy}</span>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}
