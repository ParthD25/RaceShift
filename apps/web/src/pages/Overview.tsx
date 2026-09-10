import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { BrainCircuit, Clock3, FlaskConical, Map, TimerReset } from 'lucide-react';
import { Panel } from '../components/Panel';
import { SourceBadge, sourceKindFor } from '../components/SourceBadge';
import { useForecast } from '../context/ForecastContext';
import { api, fmtDelta, fmtLap, fmtNumber } from '../lib/api';
import { useApi } from '../lib/useApi';

// Every panel on this page is read from the local API: the forecast and backtest the user ran,
// the committed experiment reports and the artifacts on disk. Nothing is illustrative.
export default function Overview() {
  const { result, backtest } = useForecast();
  const experiments = useApi(() => api.experiments());
  const reports = useApi(() => api.reports());
  const runs = [...(experiments.data?.experiments ?? [])].sort((a, b) => Number(a.is_synthetic) - Number(b.is_synthetic)).slice(0, 8);
  const deltaVsLast = result ? result.predicted_next_lap_s - result.last_lap_time_s : null;

  const mainReport = useMemo(() => (reports.data?.reports ?? []).find(r => r.name === 'f1_2025h2' && r.breakdowns) ?? (reports.data?.reports ?? []).find(r => r.breakdowns) ?? null, [reports.data]);
  const [dimension, setDimension] = useState<'circuit' | 'team' | 'compound'>('circuit');
  const breakdown = mainReport?.breakdowns?.[dimension] ?? null;
  const ffrModel = mainReport?.models.find(m => m === 'FFR-M') ?? mainReport?.models.find(m => m.startsWith('FFR')) ?? null;
  const treeModel = mainReport?.models.includes('hist_gradient_boosting') ? 'hist_gradient_boosting' : null;
  const naiveModel = mainReport?.models.includes('previous_lap') ? 'previous_lap' : null;
  const breakdownRows = breakdown ? Object.entries(breakdown).sort((a, b) => b[1].rows - a[1].rows).slice(0, 12) : [];
  const maxMae = breakdownRows.reduce((m, [, r]) => Math.max(m, ...[ffrModel, treeModel, naiveModel].filter(Boolean).map(k => r[k as string] ?? 0)), 0) || 1;

  return (
    <div className="dashboard-grid">
      <Panel title="Next Lap Forecast" icon={<TimerReset size={17} />} className="forecast-card" action={result ? <SourceBadge kind={sourceKindFor(result.is_synthetic)} /> : <SourceBadge kind="local" text="No forecast yet" />}>
        {result ? (
          <>
            <div className="forecast-main">
              <div>
                <div className="forecast-time">{fmtLap(result.predicted_next_lap_s)}</div>
                <div className="forecast-vs"><span className={deltaVsLast != null && deltaVsLast <= 0 ? 'good' : 'bad'}>{deltaVsLast != null && deltaVsLast <= 0 ? '▼' : '▲'} {fmtDelta(deltaVsLast)}</span><small>vs. lap {result.lap_number_completed}</small></div>
              </div>
              <div className="confidence-chip">±{fmtNumber((result.upper_80_s - result.lower_80_s) / 2)}s · 80%</div>
            </div>
            <div className="interval-label"><span>80% prediction interval</span><span>{fmtLap(result.lower_80_s)} to {fmtLap(result.upper_80_s)}</span></div>
            <div className="forecast-meta">
              <div><strong>{result.driver}</strong><span>{result.event} {result.season}</span></div>
              <div><strong>{fmtLap(result.rolling5_baseline_s)}</strong><span>Rolling-5 baseline</span></div>
              <div><strong>{result.artifact}</strong><span>Model</span></div>
            </div>
          </>
        ) : (
          <div className="empty-inline tall">No forecast has been run in this session. Open <Link to="/forecast">Forecast</Link>, pick a local dataset and artifact, and run it. The shipped 2025 season file gives a real lap on the first run.</div>
        )}
      </Panel>

      <Panel title="How did the model do on the laps that were driven?" icon={<Clock3 size={17} />} className="circuit-card" action={backtest ? <SourceBadge kind={sourceKindFor(backtest.is_synthetic)} text={`${backtest.summary.rows} lap pairs`} /> : <SourceBadge kind="local" text="Backtest" />}>
        {backtest ? (
          <>
            <div className="forecast-meta">
              <div><strong>{fmtNumber(backtest.summary.mae_s)} s</strong><span>Mean abs. error, model</span></div>
              <div><strong>{fmtNumber(backtest.summary.previous_lap_mae_s)} s</strong><span>Mean abs. error, repeat last lap</span></div>
              <div><strong>{(backtest.summary.interval80_coverage * 100).toFixed(0)}%</strong><span>Laps inside the 80% interval</span></div>
            </div>
            <div className="lap-table" style={{ marginTop: 12 }}>
              <div className="lap-head"><span>Lap</span><span>Actual</span><span>Predicted</span><span>Error</span></div>
              {backtest.laps.slice(-8).map(l => (
                <div className="lap-row" key={l.next_lap_number}><span>{l.next_lap_number}</span><span>{fmtLap(l.actual_next_lap_s)}</span><span>{fmtLap(l.predicted_next_lap_s)}</span><span className={l.abs_error_s <= 0.5 ? 'good' : 'bad'}>{fmtDelta(l.error_s)}</span></div>
              ))}
            </div>
            <div className="model-note">{backtest.driver} · {backtest.event} {backtest.season}. Each row predicts lap N+1 from lap N without seeing it; {backtest.skipped_laps} of {backtest.laps_driven} laps were pit, safety-car or otherwise invalid pairs and are skipped, as in training.</div>
          </>
        ) : (
          <div className="empty-inline tall">Run a forecast and the last ten completed laps of that driver are scored here: predicted vs actual, plus the naive "repeat the last lap" error for comparison.</div>
        )}
      </Panel>

      <Panel title="Where the models struggle" icon={<Map size={17} />} className="explain-card" action={mainReport ? <SourceBadge kind="real" text={`reports/${mainReport.name}`} /> : <SourceBadge kind="offline" text="No report" />}>
        {reports.error && <div className="error-box">{reports.error}</div>}
        {!reports.error && !mainReport && !reports.loading && <div className="empty-inline">No committed breakdown report found under reports/.</div>}
        {mainReport && (
          <>
            <div className="segmented" style={{ marginBottom: 10 }}>
              {(['circuit', 'team', 'compound'] as const).map(d => <button key={d} className={d === dimension ? 'selected' : ''} onClick={() => setDimension(d)}>{d}</button>)}
            </div>
            {breakdownRows.map(([label, row]) => (
              <div className="bar-row" key={label} title={`${label}: ${ffrModel ? `${ffrModel} ${fmtNumber(row[ffrModel])} s` : ''}${treeModel ? `, trees ${fmtNumber(row[treeModel])} s` : ''}${naiveModel ? `, previous lap ${fmtNumber(row[naiveModel])} s` : ''} over ${row.rows} test laps`}>
                <span>{label}</span>
                <div>
                  {ffrModel && <div className="bar-track" title={`${ffrModel} ${fmtNumber(row[ffrModel])} s`}><span style={{ width: `${(row[ffrModel] / maxMae) * 100}%` }} /></div>}
                  {treeModel && <div className="bar-track alt" style={{ marginTop: 3 }} title={`gradient-boosted trees ${fmtNumber(row[treeModel])} s`}><span style={{ width: `${(row[treeModel] / maxMae) * 100}%` }} /></div>}
                </div>
                <span>{ffrModel ? fmtNumber(row[ffrModel]) : '—'} s</span>
              </div>
            ))}
            <div className="model-note">Test MAE in seconds per {dimension} ({mainReport.title ?? mainReport.name}). Blue: {ffrModel ?? 'FFR'}; red: gradient-boosted trees. Hover a bar for the exact value and lap count.</div>
          </>
        )}
      </Panel>

      <Panel title="Model Experiments" icon={<FlaskConical size={17} />} className="experiments-card" action={experiments.error ? <SourceBadge kind="offline" /> : <SourceBadge kind="local" text="Read from artifacts/" />}>
        {experiments.error && <div className="error-box">{experiments.error}</div>}
        {!experiments.error && !experiments.loading && runs.length === 0 && <div className="empty-inline">No runs in artifacts/ yet.</div>}
        {runs.length > 0 && (
          <div className="experiment-table">
            <div className="experiment-head"><span>Run</span><span>Method</span><span>Test split</span><span>MAE ↓</span><span>RMSE ↓</span><span>Coverage</span><span>Source</span></div>
            {runs.map(e => (
              <div className="experiment-row" key={e.run}>
                <span><strong>{e.run}</strong></span>
                <span>{e.method ?? '—'}</span>
                <span>{e.split?.test ?? '—'}</span>
                <span>{fmtNumber(e.test?.mae_s)}</span>
                <span>{fmtNumber(e.test?.rmse_s)}</span>
                <span>{e.test?.interval80_coverage != null ? `${(e.test.interval80_coverage * 100).toFixed(0)}%` : '—'}</span>
                <span><SourceBadge kind={sourceKindFor(e.is_synthetic)} text={e.is_synthetic ? 'Synthetic' : 'Real'} /></span>
              </div>
            ))}
          </div>
        )}
        <div className="model-note"><BrainCircuit size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />Lower MAE is better. The synthetic fixture is easier than any real race, so synthetic rows are listed last and are never Formula 1 results. Full tables with resource costs: <Link to="/experiments">Experiments</Link>.</div>
      </Panel>
    </div>
  );
}
