import { NavLink, Outlet } from 'react-router-dom';
import {
  BrainCircuit, Database, FlaskConical, Gauge, GitCompareArrows, Settings, Target
} from 'lucide-react';
import { Logo } from './Logo';
import { SourceBadge, sourceKindFor } from './SourceBadge';
import { useForecast } from '../context/ForecastContext';
import { api } from '../lib/api';
import { useApi } from '../lib/useApi';

// Pages backed by the local API and real files.
const nav = [
  ['/', 'Overview', Gauge],
  ['/forecast', 'Forecast', Target],
  ['/compare', 'Compare Drivers', GitCompareArrows],
  ['/experiments', 'Experiments', FlaskConical],
  ['/datasets', 'Datasets', Database],
  ['/models', 'Models', BrainCircuit]
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
  const { active } = useForecast();
  const health = useApi(() => api.health());
  const runtime = useApi(() => api.runtime());
  const online = Boolean(health.data) && !health.error;
  const apiBind = runtime.data?.api_bind ?? (online ? 'local API' : 'not reachable');

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
          <NavLink to="/settings" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <Settings size={18} strokeWidth={1.8} />
            <span>Settings</span>
          </NavLink>
        </nav>
        <div className="sidebar-spacer" />
        <div className="dataset-mini">
          <div className="dataset-mini-title"><Database size={16} /> Local API</div>
          <div className="dataset-mini-row"><span>{apiBind}</span><span className={online ? 'ok-dot' : 'off-dot'} /></div>
          <div className="dataset-mini-row"><span>Demo artifact</span><span className={health.data?.demo_artifact_ready ? 'ok-dot' : 'off-dot'} /></div>
          <div className="dataset-mini-row" title="RaceShift reads local files only. There is no live-timing connection and none is planned for this release."><span>Live timing</span><span className="off-dot" /></div>
          <div className="storage-copy">{online ? `Offline-local mode · v${health.data?.version}` : health.error ?? 'Connecting…'}</div>
        </div>
        <div className="sidebar-version">v0.4.0</div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="context-row">
            <ContextBox label="Event" value={active ? `${active.event} · ${active.season}` : 'No forecast yet'} />
            <ContextBox label="Session" value={active ? active.session : '—'} />
            <ContextBox label="Driver" value={active ? active.driver : '—'} wide />
            <div className="model-status">
              <span className={online ? 'status-dot' : 'off-dot'} />
              <span><small>Model</small><strong>{active ? active.artifact : runtime.data?.default_artifact.id ?? '—'}</strong></span>
              {active
                ? <SourceBadge kind={sourceKindFor(active.is_synthetic || Boolean(active.data_is_synthetic))} text={active.data_is_synthetic ? (active.is_synthetic ? 'Synthetic' : 'Real model · synthetic laps') : (active.is_synthetic ? 'Synthetic model · real laps' : 'Real')} />
                : runtime.data
                  ? <SourceBadge kind={sourceKindFor(runtime.data.default_artifact.is_synthetic)} text={runtime.data.default_artifact.is_synthetic ? 'Synthetic demo' : 'Real data'} />
                  : null}
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
