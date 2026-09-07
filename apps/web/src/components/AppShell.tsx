import { NavLink, Outlet } from 'react-router-dom';
import {
  Activity, BarChart3, BrainCircuit, ChevronDown, Database, FlaskConical, Gauge,
  GitCompareArrows, Moon, Settings, SlidersHorizontal, Sparkles, Target, Waypoints
} from 'lucide-react';
import { Logo } from './Logo';

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

function SelectBox({ label, value, wide = false }: { label: string; value: string; wide?: boolean }) {
  return (
    <button className={`context-select ${wide ? 'wide' : ''}`} type="button">
      <span className="context-label">{label}</span>
      <span className="context-value">{value}</span>
      <ChevronDown size={14} aria-hidden="true" />
    </button>
  );
}

export function AppShell() {
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
          <div className="dataset-mini-title"><Database size={16} /> Dataset status</div>
          <div className="dataset-mini-row"><span>F1 core</span><span className="ok-dot" /></div>
          <div className="dataset-mini-row"><span>Weather</span><span className="ok-dot" /></div>
          <div className="dataset-mini-row"><span>Circuits</span><span className="ok-dot" /></div>
          <div className="storage-bar"><span style={{ width: '63%' }} /></div>
          <div className="storage-copy">Local demo fixture</div>
        </div>
        <div className="sidebar-version">v0.4.0</div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="context-row">
            <SelectBox label="Event" value="Monza · Italy" />
            <SelectBox label="Session" value="Race" />
            <SelectBox label="Driver" value="#16 Charles Leclerc" wide />
            <button className="model-status" type="button">
              <span className="status-dot" />
              <span><small>Model</small><strong>RaceShift FFR · local demo</strong></span>
              <ChevronDown size={14} />
            </button>
          </div>
          <div className="topbar-actions">
            <button className="icon-btn" aria-label="Theme"><Moon size={17} /></button>
            <button className="icon-btn" aria-label="Controls"><SlidersHorizontal size={17} /></button>
          </div>
        </header>
        <main className="content"><Outlet /></main>
      </div>
    </div>
  );
}
