import { useEffect, useMemo, useState } from 'react';
import { GitCompareArrows, Play } from 'lucide-react';
import { Panel } from '../components/Panel';
import { SourceBadge, sourceKindFor } from '../components/SourceBadge';
import { api, errorMessage, fmtDelta, fmtLap, fmtNumber, type ArtifactEntry, type BacktestResult, type ImportFile, type ImportSummary } from '../lib/api';
import { useApi } from '../lib/useApi';

// Two drivers, same session, same model: side-by-side backtests on laps that were really driven,
// plus each model's per-driver test error from the committed experiment metrics.
export default function Compare() {
  const datasets = useApi(() => api.datasets());
  const models = useApi(() => api.models());
  const driverReports = useApi(() => api.driverReports());

  const [file, setFile] = useState('');
  const [artifact, setArtifact] = useState('');
  const [left, setLeft] = useState('');
  const [right, setRight] = useState('');
  const [summary, setSummary] = useState<ImportSummary | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<[BacktestResult, BacktestResult] | null>(null);

  const imports: ImportFile[] = datasets.data?.imports ?? [];
  const artifacts: ArtifactEntry[] = useMemo(() => (models.data?.artifacts ?? []).filter(a => a.ready && a.role === 'primary-trainable'), [models.data]);

  useEffect(() => {
    if (!file && imports.length) setFile(imports.find(i => i.name === 'f1_2025_season.parquet')?.name ?? imports[0].name);
  }, [imports, file]);
  useEffect(() => {
    if (!artifact && models.data) setArtifact(artifacts.find(a => a.is_default)?.id ?? artifacts[0]?.id ?? '');
  }, [models.data, artifacts, artifact]);
  useEffect(() => {
    if (!file) return;
    let cancelled = false;
    setSummary(null);
    setResults(null);
    api.importSummary(file).then(s => {
      if (cancelled) return;
      setSummary(s);
      const drivers = s.latest_session_drivers ?? [];
      setLeft(drivers[0] ?? '');
      setRight(drivers[1] ?? drivers[0] ?? '');
    }).catch(err => { if (!cancelled) setError(errorMessage(err)); });
    return () => { cancelled = true; };
  }, [file]);

  const drivers = summary?.latest_session_drivers ?? [];
  const canRun = Boolean(file && artifact && left && right) && !running;

  async function run() {
    setRunning(true);
    setError(null);
    try {
      const [a, b] = await Promise.all([
        api.backtest({ file, driver: left, artifact, laps: 15 }),
        api.backtest({ file, driver: right, artifact, laps: 15 })
      ]);
      setResults([a, b]);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setRunning(false);
    }
  }

  const realReports = (driverReports.data?.artifacts ?? []).filter(r => !r.is_synthetic);

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div><h1>Compare Drivers</h1><p>Same session, two drivers: their actual pace over the last laps side by side, how the next-lap forecast did on each, and every model's per-driver error on the 2025 test rounds.</p></div>
        <button className="primary-btn" onClick={run} disabled={!canRun}><Play size={15} />{running ? 'Running…' : 'Backtest both'}</button>
      </div>

      <Panel title="Selection" icon={<GitCompareArrows size={17} />} action={<SourceBadge kind="local" />}>
        <div className="form-grid">
          <label className="field"><span>Dataset (data/imports)</span>
            <select value={file} onChange={e => setFile(e.target.value)} disabled={!imports.length}>
              {imports.map(i => <option key={i.name} value={i.name}>{i.name}</option>)}
            </select>
          </label>
          <label className="field"><span>Driver A</span>
            <select value={left} onChange={e => setLeft(e.target.value)} disabled={!drivers.length}>{drivers.map(d => <option key={d} value={d}>{d}</option>)}</select>
          </label>
          <label className="field"><span>Driver B</span>
            <select value={right} onChange={e => setRight(e.target.value)} disabled={!drivers.length}>{drivers.map(d => <option key={d} value={d}>{d}</option>)}</select>
          </label>
          <label className="field"><span>Model artifact</span>
            <select value={artifact} onChange={e => setArtifact(e.target.value)} disabled={!artifacts.length}>
              {artifacts.map(a => <option key={a.id} value={a.id}>{a.id}{a.is_synthetic ? ' (synthetic demo)' : ''}</option>)}
            </select>
          </label>
        </div>
        {summary && <p className="prose small">Latest session in this file: {summary.latest_session ? `${summary.latest_session.season} · ${summary.latest_session.event} · ${summary.latest_session.session}` : '—'}. Backtests use the last 15 completed lap pairs of each driver.</p>}
        {error && <div className="error-box">{error}</div>}
      </Panel>

      {results && (
        <div className="compare-grid">
          {results.map(r => (
            <Panel key={r.driver} title={`${r.driver} · ${r.event} ${r.season}`} icon={<GitCompareArrows size={17} />} action={<SourceBadge kind={sourceKindFor(r.is_synthetic)} text={`${r.summary.rows} lap pairs`} />}>
              <div className="forecast-meta">
                <div><strong>{fmtLap(r.laps.reduce((acc, l) => acc + l.actual_next_lap_s, 0) / Math.max(1, r.laps.length))}</strong><span>Average actual lap, these {r.laps.length} laps</span></div>
                <div><strong>{fmtLap(Math.min(...r.laps.map(l => l.actual_next_lap_s)))}</strong><span>Best actual lap</span></div>
                <div><strong>{r.laps[r.laps.length - 1]?.position != null ? `P${r.laps[r.laps.length - 1].position!.toFixed(0)}` : '—'}</strong><span>Position at the last of these laps</span></div>
              </div>
              <div className="forecast-meta" style={{ marginTop: 8 }}>
                <div><strong>{fmtNumber(r.summary.mae_s)} s</strong><span>Model MAE</span></div>
                <div><strong>{fmtNumber(r.summary.previous_lap_mae_s)} s</strong><span>Repeat-last-lap MAE</span></div>
                <div><strong>{(r.summary.interval80_coverage * 100).toFixed(0)}%</strong><span>Inside 80% interval</span></div>
              </div>
              <div className="wide-table backtest-table" style={{ marginTop: 10 }}>
                <div className="wide-head"><span>Lap</span><span>Actual</span><span>Predicted</span><span>Error</span><span>Interval</span><span>Tyre</span></div>
                {r.laps.map(l => (
                  <div className="wide-row" key={l.next_lap_number}>
                    <span>{l.next_lap_number}</span>
                    <span>{fmtLap(l.actual_next_lap_s)}</span>
                    <span>{fmtLap(l.predicted_next_lap_s)}</span>
                    <span className={l.abs_error_s <= 0.5 ? 'good' : 'bad'}>{fmtDelta(l.error_s)}</span>
                    <span>{l.within_interval ? 'inside' : 'outside'}</span>
                    <span>{l.compound ?? '—'}{l.tyre_life != null ? ` · ${l.tyre_life.toFixed(0)}` : ''}</span>
                  </div>
                ))}
              </div>
              <p className="prose small">{r.skipped_laps} of {r.laps_driven} laps had no valid adjacent pair (pit, safety car, red flag, deleted) and are not scored.</p>
            </Panel>
          ))}
        </div>
      )}

      <Panel title="Per-driver test error of the committed models (2025 test rounds)" icon={<GitCompareArrows size={17} />} action={realReports.length ? <SourceBadge kind="real" text="metrics.json" /> : <SourceBadge kind="offline" text="No real artifact" />}>
        {driverReports.error && <div className="error-box">{driverReports.error}</div>}
        {realReports.length === 0 && !driverReports.loading && <div className="empty-inline">No real-data artifact with per-driver metrics is present under artifacts/.</div>}
        {realReports.length > 0 && (
          <div className="wide-table backtest-table">
            <div className="wide-head"><span>Driver</span>{realReports.map(r => <span key={r.artifact}>{r.name} MAE</span>)}<span>Test laps</span><span /><span /></div>
            {Object.keys(realReports[0].test_by_driver).sort().map(driver => (
              <div className={`wide-row ${driver === left || driver === right ? 'highlight' : ''}`} key={driver}>
                <span><strong>{driver}</strong></span>
                {realReports.map(r => <span key={r.artifact}>{fmtNumber(r.test_by_driver[driver]?.mae_s)} s</span>)}
                <span>{realReports[0].test_by_driver[driver]?.rows ?? '—'}</span><span /><span />
              </div>
            ))}
          </div>
        )}
        <p className="prose small">Mean absolute error of the next-lap forecast per driver on the held-out 2025 rounds. Differences between drivers mostly reflect how disrupted their races were, not driver skill.</p>
      </Panel>
    </div>
  );
}
