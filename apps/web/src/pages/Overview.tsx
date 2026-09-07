import { Link } from 'react-router-dom';
import { Activity, BrainCircuit, Clock3, Gauge, Thermometer, TimerReset } from 'lucide-react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Panel } from '../components/Panel';
import { CircuitMap } from '../components/CircuitMap';
import { SourceBadge, sourceKindFor } from '../components/SourceBadge';
import { useForecast } from '../context/ForecastContext';
import { factors, recentLaps, telemetry } from '../data/mock';
import { api, fmtDelta, fmtLap, fmtNumber } from '../lib/api';
import { useApi } from '../lib/useApi';

const fmtFixtureDelta = (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(3)}`;

export default function Overview() {
  const { result } = useForecast();
  const experiments = useApi(() => api.experiments());
  const runs = (experiments.data?.experiments ?? []).slice(0, 6);
  const deltaVsLast = result ? result.predicted_next_lap_s - result.last_lap_time_s : null;

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
              <div><strong>{fmtNumber(result.layer_disagreement_s)}s</strong><span>Layer disagreement</span></div>
            </div>
          </>
        ) : (
          <div className="empty-inline tall">No forecast has been run in this session. Open <Link to="/forecast">Forecast</Link>, pick a local dataset and artifact, and run it. Nothing on this card is invented.</div>
        )}
      </Panel>

      <Panel title="Circuit Analysis" icon={<Gauge size={17} />} action={<SourceBadge kind="fixture" />} className="circuit-card">
        <CircuitMap />
      </Panel>

      <Panel title="Model Explanation" icon={<BrainCircuit size={17} />} className="explain-card" action={<SourceBadge kind="fixture" />}>
        <div className="factor-list">
          {factors.map(f => <div className="factor" key={f.label}><div className="factor-row"><span>{f.label}</span><span>{f.value}%</span></div><div className="factor-bar"><span style={{ width: `${f.value * 3.5}%` }} /></div></div>)}
        </div>
        <div className="model-note">Illustrative layout for a future attribution view. RaceShift FFR does not yet export per-feature explanations; these bars are not model output.</div>
      </Panel>

      <Panel title="Telemetry & Pace Comparison" icon={<Activity size={17} />} className="telemetry-card" action={<SourceBadge kind="fixture" />}>
        <div className="chart-legend"><span className="speed">Speed</span><span className="throttle">Throttle</span><span className="brake">Brake</span></div>
        <div className="telemetry-chart">
          <ResponsiveContainer width="100%" height={245}>
            <LineChart data={telemetry} margin={{ top: 8, right: 12, bottom: 0, left: -16 }}>
              <CartesianGrid stroke="#1e252c" vertical={false} />
              <XAxis dataKey="distance" tick={{ fill: '#7e8995', fontSize: 10 }} axisLine={{ stroke: '#2a333c' }} tickLine={false} />
              <YAxis domain={[0, 350]} tick={{ fill: '#7e8995', fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#10161c', border: '1px solid #28313a', borderRadius: 8, fontSize: 11 }} />
              <Line type="monotone" dataKey="speed" stroke="#dce5ed" dot={false} strokeWidth={1.7} />
              <Line type="monotone" dataKey="throttle" stroke="#42d27a" dot={false} strokeWidth={1.45} />
              <Line type="monotone" dataKey="brake" stroke="#ef4b50" dot={false} strokeWidth={1.35} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <Panel title="Recent Laps" icon={<Clock3 size={17} />} className="recent-card" action={<SourceBadge kind="fixture" />}>
        <div className="lap-table"><div className="lap-head"><span>Lap</span><span>Actual</span><span>Predicted</span><span>Delta</span></div>{recentLaps.map(l => <div className="lap-row" key={l.lap}><span>{l.lap}</span><span>{l.actual}</span><span>{l.predicted}</span><span className={l.delta <= 0 ? 'good' : 'bad'}>{fmtFixtureDelta(l.delta)}</span></div>)}</div>
      </Panel>

      <Panel title="Model Experiments" icon={<Thermometer size={17} />} className="experiments-card" action={experiments.error ? <SourceBadge kind="offline" /> : <SourceBadge kind="local" text="Read from artifacts/" />}>
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
      </Panel>
    </div>
  );
}
