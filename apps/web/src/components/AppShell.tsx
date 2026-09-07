import { NavLink, Outlet } from 'react-router-dom';
import {
  Activity, BrainCircuit, Database, FlaskConical, Gauge, GitCompareArrows, Settings, Sparkles, Target
} from 'lucide-react';
import { Logo } from './Logo';
import { SourceBadge, sourceKindFor } from './SourceBadge';
import { useForecast } from '../context/ForecastContext';
import { api } from '../lib/api';
import { useApi } from '../lib/useApi';

const nav = [
  ['/', 'Overview', Gauge],
  ['/forecast', 'Forecast', Target],
  ['/telemetry', 'Telemetry', Activity],
  ['/experiments', 'Experiments', FlaskConical],
  ['/datasets', 'Datasets', Database],
  ['/models', 'Models', BrainCircuit]
] as const;

const secondary = [
  ['/compare', 'Compare Drivers', GitCompareArrows],
  ['/strategy', 'Strategy Insights', Sparkles],
  ['/settings', 'Settings', Settings]
] as const;

function ContextBox({ label, value, wide = false }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={`context-select ${wide ? 'wide' : ''}`}>
      <span className="context-label">{label}</span>
      <span className="context-value">{value}</span>
    </div>
  );
}

export function AppShell() {
  const { result } = useForecast();
  const health = useApi(() => api.health());
  const online = Boolean(health.data) && !health.error;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Logo />
        <nav className="nav-list" aria-label="Primary navigation">
          {nav.map(([to, label, Icon]) => (
            <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
              <Icon size={18} strokeWidth={1.8} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="nav-divider" />
        <nav className="nav-list secondary" aria-label="Secondary navigation">
          {secondary.map(([to, label, Icon]) => (
            <NavLink key={to} to={to} className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
              <Icon size={18} strokeWidth={1.8} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="dataset-mini">
          <div className="dataset-mini-title"><Database size={16} /> Local API</div>
          <div className="dataset-mini-row"><span>127.0.0.1:8000</span><span className={online ? 'ok-dot' : 'off-dot'} /></div>
          <div className="dataset-mini-row"><span>Demo artifact</span><span className={health.data?.demo_artifact_ready ? 'ok-dot' : 'off-dot'} /></div>
          <div className="dataset-mini-row"><span>Live timing</span><span className="off-dot" title="Not configured. RaceShift is offline-first." /></div>
          <div className="storage-copy">{online ? `Offline-local mode · v${health.data?.version}` : health.error ?? 'Connecting…'}</div>
        </div>
        <div className="sidebar-version">v0.4.0</div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="context-row">
            <ContextBox label="Event" value={result ? `${result.event} · ${result.season}` : 'No forecast yet'} />
            <ContextBox label="Session" value={result ? result.session : '—'} />
            <ContextBox label="Driver" value={result ? result.driver : '—'} wide />
            <div className="model-status">
              <span className={online ? 'status-dot' : 'off-dot'} />
              <span><small>Model</small><strong>{result ? result.artifact : 'raceshift_ffr_demo'}</strong></span>
              {result ? <SourceBadge kind={sourceKindFor(result.is_synthetic)} text={result.is_synthetic ? 'Synthetic' : 'Real'} /> : <SourceBadge kind="synthetic" text="Synthetic demo" />}
            </div>
          </div>
          <div className="topbar-actions">
            {online ? <SourceBadge kind="local" text="Local API online" /> : <SourceBadge kind="offline" />}
          </div>
        </header>
        <main className="content"><Outlet /></main>
      </div>
    </div>
  );
}
