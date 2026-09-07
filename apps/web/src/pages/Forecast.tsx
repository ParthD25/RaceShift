import { useEffect, useMemo, useState } from 'react';
import { BrainCircuit, Play, Target, TrendingDown } from 'lucide-react';
import { Panel } from '../components/Panel';
import { SourceBadge, sourceKindFor } from '../components/SourceBadge';
import { useForecast } from '../context/ForecastContext';
import { api, errorMessage, fmtDelta, fmtLap, fmtNumber, type ArtifactEntry, type ImportFile, type ImportSummary } from '../lib/api';
import { useApi } from '../lib/useApi';

export default function Forecast() {
  const datasets = useApi(() => api.datasets());
  const models = useApi(() => api.models());
  const { result, setResult } = useForecast();

  const [file, setFile] = useState('');
  const [driver, setDriver] = useState('');
  const [artifact, setArtifact] = useState('');
  const [summary, setSummary] = useState<ImportSummary | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const imports: ImportFile[] = datasets.data?.imports ?? [];
  const artifacts: ArtifactEntry[] = useMemo(() => (models.data?.artifacts ?? []).filter(a => a.ready && a.role === 'primary-trainable'), [models.data]);

  useEffect(() => {
    if (!file && imports.length) setFile(imports.find(i => i.name === 'synthetic_fixture.csv')?.name ?? imports[0].name);
  }, [imports, file]);
  useEffect(() => {
    if (!artifact && models.data) setArtifact(artifacts.find(a => a.is_default)?.id ?? artifacts[0]?.id ?? '');
  }, [models.data, artifacts, artifact]);
  useEffect(() => {
    if (!file) return;
    let cancelled = false;
    setSummary(null);
    setSummaryError(null);
    api.importSummary(file)
      .then(s => { if (!cancelled) { setSummary(s); setDriver(''); } })
      .catch(err => { if (!cancelled) setSummaryError(errorMessage(err)); });
    return () => { cancelled = true; };
  }, [file]);

  const drivers = summary?.latest_session_drivers ?? [];
  const selectedArtifact = artifacts.find(a => a.id === artifact);
  const canRun = Boolean(file && artifact) && !running && (summary?.missing_required_columns.length ?? 0) === 0;

  async function run() {
    setRunning(true);
    setError(null);
    try {
      setResult(await api.forecastLatest({ file, driver: driver || null, artifact: artifact || null }));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setRunning(false);
    }
  }

  const deltaVsLast = result ? result.predicted_next_lap_s - result.last_lap_time_s : null;
  const intervalPct = result ? intervalGeometry(result.lower_80_s, result.upper_80_s, result.predicted_next_lap_s, result.last_lap_time_s) : null;

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div><h1>Forecast</h1><p>Predict lap N+1 from information available at the end of lap N, using the local API and a saved artifact.</p></div>
        <button className="primary-btn" onClick={run} disabled={!canRun}><Play size={15} />{running ? 'Running…' : 'Run forecast'}</button>
      </div>

      <div className="two-col">
        <Panel title="Forecast inputs" icon={<Target size={17} />} action={datasets.error || models.error ? <SourceBadge kind="offline" /> : <SourceBadge kind="local" />}>
          {(datasets.error || models.error) && <div className="error-box">{datasets.error ?? models.error}</div>}
          <div className="form-grid">
            <label className="field"><span>Dataset (data/imports)</span>
              <select value={file} onChange={e => setFile(e.target.value)} disabled={!imports.length}>
                {!imports.length && <option value="">No CSV/Parquet files in data/imports</option>}
                {imports.map(i => <option key={i.name} value={i.name}>{i.name}</option>)}
              </select>
            </label>
            <label className="field"><span>Driver (latest session)</span>
              <select value={driver} onChange={e => setDriver(e.target.value)} disabled={!drivers.length}>
                <option value="">Auto: most completed laps</option>
                {drivers.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </label>
            <label className="field"><span>Model artifact</span>
              <select value={artifact} onChange={e => setArtifact(e.target.value)} disabled={!artifacts.length}>
                {!artifacts.length && <option value="">No complete FFR artifact in artifacts/</option>}
                {artifacts.map(a => <option key={a.id} value={a.id}>{a.id}{a.is_synthetic ? ' (synthetic demo)' : ''}</option>)}
              </select>
            </label>
          </div>
          {summaryError && <div className="error-box">{summaryError}</div>}
          {summary && (
            <div className="detail-list compact">
              <div><span>Rows</span><strong>{summary.rows.toLocaleString()}</strong></div>
              <div><span>Seasons</span><strong>{summary.seasons?.join(', ') ?? '—'}</strong></div>
              <div><span>Latest session</span><strong>{summary.latest_session ? `${summary.latest_session.season} · ${summary.latest_session.event} · ${summary.latest_session.session}` : '—'}</strong></div>
              <div><span>Provenance</span><strong>{summary.is_synthetic ? 'Synthetic fixture' : 'User-supplied data'}</strong></div>
              {summary.missing_required_columns.length > 0 && <div><span>Missing columns</span><strong className="bad">{summary.missing_required_columns.join(', ')}</strong></div>}
            </div>
          )}
          {selectedArtifact && (
            <div className="callout">
              <BrainCircuit size={18} />
              <div>
                <strong>{selectedArtifact.name} · {selectedArtifact.id} <SourceBadge kind={sourceKindFor(selectedArtifact.is_synthetic)} /></strong>
                <p>{selectedArtifact.is_synthetic ? 'Trained on the synthetic engineering fixture. Use it to verify the workflow, not to judge Formula 1 accuracy.' : `Trained on ${selectedArtifact.data_source}.`}</p>
              </div>
            </div>
          )}
        </Panel>

        <Panel title="Forecast contract" icon={<BrainCircuit size={17} />}>
          <div className="detail-list">
            <div><span>Target</span><strong>Lap N+1 residual vs rolling-5 median</strong></div>
            <div><span>Context</span><strong>Current lap + 4 lagged laps</strong></div>
            <div><span>Primary model</span><strong>RaceShift FFR (no global backprop)</strong></div>
            <div><span>Calibration</span><strong>80% validation-residual interval</strong></div>
            <div><span>Validation</span><strong>Season-forward split</strong></div>
          </div>
          <div className="callout"><TrendingDown size={18} /><div><strong>No target leakage</strong><p>Inputs are restricted to information available at the end of lap N. Historical priors use earlier events only.</p></div></div>
        </Panel>
      </div>

      <Panel title="Next-lap result" icon={<Target size={17} />} action={result ? <SourceBadge kind={sourceKindFor(result.is_synthetic)} /> : null}>
        {error && <div className="error-box">{error}</div>}
        {!result && !error && <div className="empty-inline">No forecast has been run yet. Choose a dataset and press <strong>Run forecast</strong>.</div>}
        {result && (
          <div className="result-layout">
            <div>
              <div className="forecast-main">
                <div>
                  <div className="forecast-time">{fmtLap(result.predicted_next_lap_s)}</div>
                  <div className="forecast-vs"><span className={deltaVsLast != null && deltaVsLast <= 0 ? 'good' : 'bad'}>{deltaVsLast != null && deltaVsLast <= 0 ? '▼' : '▲'} {fmtDelta(deltaVsLast)}</span><small>vs. lap {result.lap_number_completed} ({fmtLap(result.last_lap_time_s)})</small></div>
                </div>
                <div className="confidence-chip">80% interval ±{fmtNumber((result.upper_80_s - result.lower_80_s) / 2)}s</div>
              </div>
              <div className="interval-label"><span>{fmtLap(result.lower_80_s)}</span><span>80% prediction interval</span><span>{fmtLap(result.upper_80_s)}</span></div>
              <div className="interval-bar">
                <span className="interval-range" style={intervalPct ? { left: `${intervalPct.left}%`, right: `${intervalPct.right}%` } : undefined} />
                <span className="interval-marker" style={intervalPct ? { left: `${intervalPct.marker}%` } : undefined} />
                {intervalPct && <span className="interval-last" style={{ left: `${intervalPct.last}%` }} title="Last completed lap" />}
              </div>
              {result.short_history && <div className="warn-box">Only {result.completed_laps_in_session} completed laps available; the model expects {result.history_laps_used}. Missing lags were imputed with training medians.</div>}
            </div>
            <div className="detail-list compact">
              <div><span>Session</span><strong>{result.season} · {result.event} · {result.session}</strong></div>
              <div><span>Driver</span><strong>{result.driver}</strong></div>
              <div><span>Laps completed</span><strong>{result.lap_number_completed} (forecast for lap {result.lap_number_completed + 1})</strong></div>
              <div><span>Rolling-5 baseline</span><strong>{fmtLap(result.rolling5_baseline_s)}</strong></div>
              <div><span>Model residual</span><strong>{fmtDelta(result.predicted_next_lap_s - result.rolling5_baseline_s)}</strong></div>
              <div><span>Layer disagreement</span><strong>{fmtNumber(result.layer_disagreement_s)}s</strong></div>
              <div><span>Artifact</span><strong>{result.artifact}</strong></div>
              <div><span>Data file</span><strong>{result.file}</strong></div>
            </div>
          </div>
        )}
      </Panel>

      {result && (
        <div className="two-col">
          <Panel title="Input context at end of lap" icon={<Target size={17} />} action={<SourceBadge kind="local" />}>
            <div className="detail-list compact">
              <div><span>Tyre</span><strong>{result.context.compound ?? '—'}{result.context.tyre_life != null ? ` · ${result.context.tyre_life.toFixed(0)} laps` : ''}{result.context.stint != null ? ` · stint ${result.context.stint.toFixed(0)}` : ''}</strong></div>
              <div><span>Consecutive clean laps</span><strong>{result.context.laps_in_segment != null ? result.context.laps_in_segment.toFixed(0) : '—'}</strong></div>
              <div><span>Position</span><strong>{result.context.position != null ? `P${result.context.position.toFixed(0)}` : '—'}</strong></div>
              <div><span>Track / air</span><strong>{fmtNumber(result.context.track_temp_c, 1)} °C / {fmtNumber(result.context.air_temp_c, 1)} °C</strong></div>
              <div><span>Humidity</span><strong>{fmtNumber(result.context.humidity_pct, 0)} %</strong></div>
              <div><span>Wind</span><strong>{fmtNumber(result.context.wind_speed_ms, 1)} m/s{result.context.wind_direction_deg != null ? ` from ${result.context.wind_direction_deg.toFixed(0)}°` : ''}</strong></div>
              <div><span>Rain</span><strong>{result.context.rainfall ?? '—'}</strong></div>
              <div><span>Gap ahead / behind</span><strong>{fmtNumber(result.context.gap_ahead_s, 1)} s / {fmtNumber(result.context.gap_behind_s, 1)} s</strong></div>
              <div><span>Team</span><strong>{result.context.team ?? '—'}</strong></div>
            </div>
          </Panel>
          <Panel title="Historical context (earlier events only)" icon={<BrainCircuit size={17} />} action={<SourceBadge kind="local" text="Prior events" />}>
            <div className="detail-list compact">
              <div><span>Driver at this circuit</span><strong>{fmtLap(result.historical_context.driver_circuit_pace_s)}</strong></div>
              <div><span>Team at this circuit</span><strong>{fmtLap(result.historical_context.team_circuit_pace_s)}</strong></div>
              <div><span>Compound at this circuit</span><strong>{fmtLap(result.historical_context.compound_circuit_pace_s)}</strong></div>
              <div><span>Driver · circuit · compound</span><strong>{fmtLap(result.historical_context.driver_circuit_compound_pace_s)}</strong></div>
              <div><span>Matched weather · compound</span><strong>{fmtLap(result.historical_context.matched_weather_compound_pace_s)}</strong></div>
              <div><span>Driver · matched weather</span><strong>{fmtLap(result.historical_context.driver_matched_weather_pace_s)}</strong></div>
            </div>
            <p className="prose small">{result.historical_context.note} Values are median lap times from previous events; the model uses them relative to the current rolling pace.</p>
          </Panel>
        </div>
      )}
    </div>
  );
}

function intervalGeometry(lower: number, upper: number, prediction: number, last: number) {
  const lo = Math.min(lower, last);
  const hi = Math.max(upper, last);
  const span = Math.max(hi - lo, 1e-6);
  const pad = 0.12;
  const scale = (v: number) => pad * 100 + ((v - lo) / span) * (1 - 2 * pad) * 100;
  return { left: scale(lower), right: 100 - scale(upper), marker: scale(prediction), last: scale(last) };
}
