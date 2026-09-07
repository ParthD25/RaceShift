import { Activity, BrainCircuit, Clock3, Gauge, Thermometer, TimerReset } from 'lucide-react';
import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Panel } from '../components/Panel';
import { CircuitMap } from '../components/CircuitMap';
import { experiments, factors, recentLaps, telemetry } from '../data/mock';

const fmtDelta = (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(3)}`;

export default function Overview() {
  return (
    <div className="dashboard-grid">
      <Panel title="Next Lap Forecast" icon={<TimerReset size={17} />} className="forecast-card">
        <div className="forecast-main">
          <div><div className="forecast-time">1:20.432</div><div className="forecast-vs"><span className="good">▼ -0.287s</span><small>vs. last lap</small></div></div>
          <div className="confidence-chip">82% confidence</div>
        </div>
        <div className="interval-label"><span>80% prediction interval</span><span>1:20.125 to 1:20.739</span></div>
        <div className="interval-bar"><span className="interval-range" /><span className="interval-marker" /></div>
        <div className="forecast-meta">
          <div><strong>Medium</strong><span>Tyre · 12 laps</span></div>
          <div><strong>32°C</strong><span>Track temp</span></div>
          <div><strong>Dry</strong><span>Track status</span></div>
        </div>
      </Panel>

      <Panel title="Circuit Analysis" icon={<Gauge size={17} />} action={<div className="segmented"><button className="selected">Time delta</button><button>Speed</button></div>} className="circuit-card">
        <CircuitMap />
      </Panel>

      <Panel title="Model Explanation" icon={<BrainCircuit size={17} />} className="explain-card">
        <div className="factor-list">
          {factors.map(f => <div className="factor" key={f.label}><div className="factor-row"><span>{f.label}</span><span>{f.value}%</span></div><div className="factor-bar"><span style={{ width: `${f.value * 3.5}%` }} /></div></div>)}
        </div>
        <div className="model-note">The forecast is faster primarily because recent pace is improving while tyre degradation remains inside the session baseline.</div>
      </Panel>

      <Panel title="Telemetry & Pace Comparison" icon={<Activity size={17} />} className="telemetry-card" action={<div className="segmented"><button className="selected">Current lap</button><button>Previous lap</button></div>}>
        <div className="chart-legend"><span className="speed">Speed</span><span className="throttle">Throttle</span><span className="brake">Brake</span><span className="rpm">RPM</span></div>
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

      <Panel title="Recent Laps" icon={<Clock3 size={17} />} className="recent-card">
        <div className="lap-table"><div className="lap-head"><span>Lap</span><span>Actual</span><span>Predicted</span><span>Delta</span></div>{recentLaps.map(l => <div className="lap-row" key={l.lap}><span>{l.lap}</span><span>{l.actual}</span><span>{l.predicted}</span><span className={l.delta <= 0 ? 'good' : 'bad'}>{fmtDelta(l.delta)}</span></div>)}</div>
      </Panel>

      <Panel title="Model Experiments" icon={<Thermometer size={17} />} className="experiments-card">
        <div className="experiment-table"><div className="experiment-head"><span>Model</span><span>Base</span><span>Method</span><span>MAE ↓</span><span>RMSE ↓</span><span>Coverage</span><span>Status</span></div>{experiments.map(e => <div className="experiment-row" key={e.name}><span><strong>{e.name}</strong></span><span>{e.base}</span><span>{e.method}</span><span>{e.mae.toFixed(3)}</span><span>{e.rmse.toFixed(3)}</span><span>{e.coverage.toFixed(2)}</span><span className="ready"><i />{e.status}</span></div>)}</div>
      </Panel>
    </div>
  );
}
