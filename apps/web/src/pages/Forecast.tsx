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
  const { result, setResult, backtest, setBacktest } = useForecast();
  const [backtestError, setBacktestError] = useState<string | null>(null);

  const [file, setFile] = useState('');
  const [driver, setDriver] = useState('');
  const [artifact, setArtifact] = useState('');
  const [summary, setSummary] = useState<ImportSummary | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [backtesting, setBacktesting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const imports: ImportFile[] = datasets.data?.imports ?? [];
  const artifacts: ArtifactEntry[] = useMemo(() => (models.data?.artifacts ?? []).filter(a => a.ready && a.role === 'primary-trainable'), [models.data]);

  useEffect(() => {
    // Prefer the shipped real season over the synthetic fixture so the first forecast is a real lap.
    if (!file && imports.length) setFile(imports.find(i => i.name === 'f1_2025_season.parquet')?.name ?? imports.find(i => !i.name.startsWith('synthetic'))?.name ?? imports[0].name);
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
  // A real-data model scoring the synthetic fixture (or the synthetic demo scoring real laps)
  // produces numbers, not evidence. Say so before and after the run.
  const domainMismatch = Boolean(summary && selectedArtifact && summary.is_synthetic !== selectedArtifact.is_synthetic);
  const resultMismatch = Boolean(result && result.is_synthetic !== (result.data_is_synthetic ?? summary?.is_synthetic ?? result.is_synthetic));
  const noPriors = Boolean(result && Object.entries(result.historical_context).every(([k, v]) => k === 'note' || v == null));
  const canRun = Boolean(file && artifact) && !running && (summary?.missing_required_columns.length ?? 0) === 0;

  async function run() {
    setRunning(true);
    setBacktesting(true);
    setError(null);
    setBacktestError(null);
    // The backtest scores the driver's last completed laps against laps that were really
    // driven; the forecast is for a lap that has not happened yet. They are independent: a
    // driver whose final lap was deleted or a pit lap cannot be forecast from, but their
    // race can still be backtested, so both run and each reports its own outcome. The API
    // builds the feature table once and serves both calls from it.
    const request = { file, driver: driver || null, artifact: artifact || null };
    const [forecast, scored] = await Promise.allSettled([api.forecastLatest(request), api.backtest({ ...request, laps: 10 })]);
    if (forecast.status === 'fulfilled') setResult(forecast.value); else { setResult(null); setError(errorMessage(forecast.reason)); }
    if (scored.status === 'fulfilled') setBacktest(scored.value); else { setBacktest(null); setBacktestError(errorMessage(scored.reason)); }
    setBacktesting(false);
    setRunning(false);
  }

  const deltaVsLast = result ? result.predicted_next_lap_s - result.last_lap_time_s : null;
  const intervalPct = result ? intervalGeometry(result.lower_80_s, result.upper_80_s, result.predicted_next_lap_s, result.last_lap_time_s) : null;

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div><h1>Forecast</h1><p>Predict lap N+1 from information available at the end of lap N, using the local API and a saved artifact.</p></div>
        <button className="primary-btn" onClick={run} disabled={!canRun}><Play size={15} />{backtesting ? 'Backtesting the last 10 laps…' : running ? 'Building features… (about 10 s the first time per file)' : 'Run forecast'}</button>
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
                <option value="">Auto: best-placed driver with the most laps (the winner)</option>
                {drivers.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </label>
            <label className="field"><span>Model artifact</span>
              <select value={artifact} onChange={e => setArtifact(e.target.value)} disabled={!artifacts.length}>
                {!artifacts.length && <option value="">No complete FFR artifact in artifacts/</option>}
                {artifacts.map(a => <option key={a.id} value={a.id}>{a.label ?? a.id}{a.is_synthetic ? ' · synthetic demo' : ''} · {a.id}</option>)}
              </select>
            </label>
          </div>
          {summaryError && <div className="error-box">{summaryError}</div>}
          {summary && (
            <div className="detail-list compact">
              <div><span>Rows</span><strong>{summary.rows.toLocaleString()}</strong></div>
              <div><span>Seasons</span><strong>{summary.seasons?.join(', ') ?? '—'}</strong></div>
              <div><span>Latest session</span><strong>{summary.latest_session ? `${summary.latest_session.season} · ${summary.latest_session.event} · ${summary.latest_session.session}` : '—'}</strong></div>
              <div><span>Provenance</span><strong>{summary.is_synthetic ? 'Synthetic fixture' : (summary.data_source && summary.data_source !== 'user_supplied' ? summary.data_source.replace(/_/g, ' ').replace(/\+/g, ' + ') : 'User-supplied data')}</strong></div>
              {summary.missing_required_columns.length > 0 && <div><span>Missing columns</span><strong className="bad">{summary.missing_required_columns.join(', ')}</strong></div>}
            </div>
          )}
          {selectedArtifact && (
            <div className="callout">
              <BrainCircuit size={18} />
              <div>
                <strong>{selectedArtifact.name} · {selectedArtifact.id} <SourceBadge kind={sourceKindFor(selectedArtifact.is_synthetic)} /></strong>
                <p>{selectedArtifact.is_synthetic ? 'Trained on the synthetic engineering fixture. Use it to verify the workflow, not to judge Formula 1 accuracy.' : `Trained on ${selectedArtifact.data_source}.`}{selectedArtifact.lap_validity_version != null ? ` Lap-validity rules v${selectedArtifact.lap_validity_version}${selectedArtifact.validity_rules_match === false ? ' (this runtime applies a newer rule set; see the warning after running)' : ''}.` : ''}</p>
              </div>
            </div>
          )}
          {domainMismatch && summary && selectedArtifact && (
            <div className="warn-box">
              {selectedArtifact.is_synthetic
                ? 'The synthetic demo model has never seen a real lap. Its forecast on this real data is a workflow check, not a prediction.'
                : 'This real-data model is being pointed at the synthetic fixture. The fixture comes from a different generating process, so the result says nothing about Formula 1 accuracy.'}
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

      {(backtest || backtestError) && (
        <Panel title={`How did it do on ${backtest?.driver ?? result?.driver ?? 'this driver'}'s last ${backtest?.summary.rows ?? 10} real laps?`} icon={<Target size={17} />} action={backtest ? <SourceBadge kind={sourceKindFor(backtest.is_synthetic)} text="Laps actually driven" /> : null}>
          {backtestError && <div className="error-box">{backtestError}</div>}
          {backtest?.lap_validity && !backtest.lap_validity.match && <div className="warn-box">Lap-validity rules differ: model v{backtest.lap_validity.artifact ?? '?'}, runtime v{backtest.lap_validity.runtime}. The laps scored here are selected by the runtime's rules, not the ones the model was trained on.</div>}
          {backtest && (
            <>
              <p className="prose">On {backtest.driver}'s last {backtest.summary.rows} real laps this model was within <strong>{fmtNumber(backtest.summary.mae_s)} s</strong> of the true next lap on average; simply repeating the last lap was within <strong>{fmtNumber(backtest.summary.previous_lap_mae_s)} s</strong>. {backtest.summary.mae_s <= backtest.summary.previous_lap_mae_s ? 'The model beat the stopwatch here.' : 'The stopwatch won on these laps. Over whole seasons the model is ahead of the stopwatch by a few hundredths of a second per lap and wins about half of all laps, which is the honest size of its edge.'}</p>
              <div className="forecast-meta">
                <div><strong>{fmtNumber(backtest.summary.mae_s)} s</strong><span>Mean abs. error, this model</span></div>
                <div><strong>{fmtNumber(backtest.summary.previous_lap_mae_s)} s</strong><span>Mean abs. error, repeat the last lap</span></div>
                <div><strong>{(backtest.summary.within_0_5s_share * 100).toFixed(0)}% · {(backtest.summary.interval80_coverage * 100).toFixed(0)}%</strong><span>Within 0.5 s · inside 80% interval</span></div>
              </div>
              <div className="wide-table backtest-table" style={{ marginTop: 10 }}>
                <div className="wide-head"><span>Lap</span><span>Actual</span><span>Predicted</span><span>Error</span><span>80% interval</span><span>Tyre</span></div>
                {backtest.laps.map(l => (
                  <div className="wide-row" key={l.next_lap_number}>
                    <span>{l.next_lap_number}</span>
                    <span>{fmtLap(l.actual_next_lap_s)}</span>
                    <span>{fmtLap(l.predicted_next_lap_s)}</span>
                    <span className={l.abs_error_s <= 0.5 ? 'good' : 'bad'}>{fmtDelta(l.error_s)}</span>
                    <span>{fmtLap(l.lower_80_s)} – {fmtLap(l.upper_80_s)}{l.within_interval ? '' : ' ✕'}</span>
                    <span>{l.compound ?? '—'}{l.tyre_life != null ? ` · ${l.tyre_life.toFixed(0)} laps` : ''}</span>
                  </div>
                ))}
              </div>
              <p className="prose small">Each row forecasts lap N+1 from what was known at the end of lap N; the model never sees the actual next lap. {backtest.skipped_laps} of {backtest.laps_driven} laps were pit, yellow-flag, safety-car, red-flag, restart or deleted laps without a valid adjacent pair and are skipped, exactly as in training.</p>
            </>
          )}
        </Panel>
      )}

      <Panel title={`Next lap forecast (lap ${result ? result.lap_number_completed + 1 : "N+1"})`} icon={<Target size={17} />} action={result ? <SourceBadge kind={resultMismatch ? 'fixture' : sourceKindFor(result.is_synthetic)} text={resultMismatch ? (result.is_synthetic ? 'Synthetic model · real laps' : 'Real model · synthetic laps') : undefined} /> : null}>
        {error && <div className="error-box">{error}</div>}
        {result?.session_warning && <div className="warn-box">{result.session_warning}</div>}
        {result?.lap_validity && !result.lap_validity.match && <div className="warn-box">This model was trained under lap-validity rules v{result.lap_validity.artifact ?? '?'} but this runtime applies v{result.lap_validity.runtime}: its inputs are built from a different set of laps than it learned on, so its published metrics do not describe this forecast. Retrain or pick a model whose rules match.</div>}
        {resultMismatch && <div className="warn-box">Model and data come from different sources ({result?.is_synthetic ? 'synthetic model on real laps' : 'real model on the synthetic fixture'}). Treat this number as a workflow check only.</div>}
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
                <span className="interval-marker" style={intervalPct ? { left: `${intervalPct.marker}%` } : undefined} title="Predicted next lap" />
                {intervalPct && <span className="interval-last" style={{ left: `${intervalPct.last}%` }} title="Last completed lap" />}
              </div>
              <div className="chart-legend small"><span className="marker-legend">Predicted next lap</span><span className="last-legend">Last completed lap</span><span className="range-legend">80% interval: sized so 8 of 10 validation laps fall inside{selectedArtifact?.test?.interval80_coverage != null ? `; on the held-out test laps it contained ${(selectedArtifact.test.interval80_coverage * 100).toFixed(0)}%` : ''}. The width is fixed per model unless the layers disagree.</span></div>
              {result.short_history && <div className="warn-box">Only {result.completed_laps_in_session} completed laps available; the model expects {result.history_laps_used}. Missing lags were imputed with training medians.</div>}
              <div className="prose small">This is the forecast for lap {result.lap_number_completed + 1}, the lap after the last one recorded in the file. If the session had already ended, that lap never happened and the number is hypothetical. Use <strong>Backtest</strong> below to see how the model did on laps that were actually driven.</div>
            </div>
            <div className="detail-list compact">
              <div><span>Session</span><strong>{result.season} · {result.event} · {result.session}</strong></div>
              <div><span>Driver</span><strong>{result.driver}</strong></div>
              <div><span>Laps completed</span><strong>{result.lap_number_completed} (forecast for lap {result.lap_number_completed + 1})</strong></div>
              <div title="Median of the last five clean laps. The model predicts how far the next lap will be from this number."><span>Rolling-5 baseline ⓘ</span><strong>{fmtLap(result.rolling5_baseline_s)}</strong></div>
              <div title="What the model adds to the rolling-5 baseline. Forecast = baseline + residual. It differs from the change versus the last lap because the last lap is usually not equal to the five-lap median."><span>Model residual ⓘ</span><strong>{fmtDelta(result.predicted_next_lap_s - result.rolling5_baseline_s)}</strong></div>
              <div title="Spread between the per-layer predictions of the Forward-Forward network. Each layer is trained on its own, so when they disagree the interval is widened."><span>Layer disagreement ⓘ</span><strong>{fmtNumber(result.layer_disagreement_s)}s</strong></div>
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
              <div><span>Driver at this circuit</span><strong className={result.historical_context.driver_circuit_pace_s == null ? "muted" : undefined}>{result.historical_context.driver_circuit_pace_s == null ? "no earlier event in this file" : fmtLap(result.historical_context.driver_circuit_pace_s)}</strong></div>
              <div><span>Team at this circuit</span><strong className={result.historical_context.team_circuit_pace_s == null ? "muted" : undefined}>{result.historical_context.team_circuit_pace_s == null ? "no earlier event in this file" : fmtLap(result.historical_context.team_circuit_pace_s)}</strong></div>
              <div><span>Compound at this circuit</span><strong className={result.historical_context.compound_circuit_pace_s == null ? "muted" : undefined}>{result.historical_context.compound_circuit_pace_s == null ? "no earlier event in this file" : fmtLap(result.historical_context.compound_circuit_pace_s)}</strong></div>
              <div><span>Driver · circuit · compound</span><strong className={result.historical_context.driver_circuit_compound_pace_s == null ? "muted" : undefined}>{result.historical_context.driver_circuit_compound_pace_s == null ? "no earlier event in this file" : fmtLap(result.historical_context.driver_circuit_compound_pace_s)}</strong></div>
              <div><span>Matched weather · compound</span><strong className={result.historical_context.matched_weather_compound_pace_s == null ? "muted" : undefined}>{result.historical_context.matched_weather_compound_pace_s == null ? "no earlier event in this file" : fmtLap(result.historical_context.matched_weather_compound_pace_s)}</strong></div>
              <div><span>Driver · matched weather</span><strong className={result.historical_context.driver_matched_weather_pace_s == null ? "muted" : undefined}>{result.historical_context.driver_matched_weather_pace_s == null ? "no earlier event in this file" : fmtLap(result.historical_context.driver_matched_weather_pace_s)}</strong></div>
              <div><span>Driver · all earlier events</span><strong className={result.historical_context.driver_overall_pace_s == null ? "muted" : undefined}>{result.historical_context.driver_overall_pace_s == null ? "no earlier event in this file" : fmtLap(result.historical_context.driver_overall_pace_s)}</strong></div>
              <div><span>Team · all earlier events</span><strong className={result.historical_context.team_overall_pace_s == null ? "muted" : undefined}>{result.historical_context.team_overall_pace_s == null ? "no earlier event in this file" : fmtLap(result.historical_context.team_overall_pace_s)}</strong></div>
            </div>
            {noPriors && <div className="warn-box">No earlier events for this driver, team or circuit exist in this file, so every prior is empty and the model falls back to training medians. Load a table with earlier rounds (the shipped f1_2025_season.parquet has the whole season) to populate them.</div>}
            {!noPriors && <p className="prose small">Priors marked "no earlier event in this file" (for example this circuit) were imputed with training medians; the published metrics were measured with 2018-2024 history present, so a single-season file runs the model with less context than it was tested with.</p>}
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
