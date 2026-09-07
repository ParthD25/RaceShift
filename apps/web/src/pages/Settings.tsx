import { CheckCircle2, Settings as SettingsIcon, ShieldCheck, XCircle } from 'lucide-react';
import { Panel } from '../components/Panel';
import { SourceBadge } from '../components/SourceBadge';
import { api } from '../lib/api';
import { useApi } from '../lib/useApi';

export default function Settings() {
  const runtime = useApi(() => api.runtime());
  const setup = useApi(() => api.setup());
  const offline = Boolean(runtime.error || setup.error);
  return (
    <div className="page-stack">
      <div className="page-heading"><div><h1>Settings</h1><p>Local runtime status and setup checklist. RaceShift stores no credentials and never claims a live connection.</p></div></div>
      <div className="two-col">
        <Panel title="Setup checklist" icon={<SettingsIcon size={17} />} action={offline ? <SourceBadge kind="offline" /> : <SourceBadge kind="local" text="Live from local API" />}>
          {setup.error && <div className="error-box">{setup.error}</div>}
          {setup.data?.checks.map(c => (
            <div className="check-row" key={c.item}>
              {c.ok ? <CheckCircle2 size={16} className="good" /> : <XCircle size={16} className="bad" />}
              <div><strong>{c.item}</strong><p>{c.detail}</p></div>
            </div>
          ))}
        </Panel>
        <Panel title="Runtime" icon={<ShieldCheck size={17} />}>
          {runtime.error && <div className="error-box">{runtime.error}</div>}
          {runtime.data && (
            <div className="detail-list compact">
              <div><span>RaceShift</span><strong>v{runtime.data.raceshift_version}</strong></div>
              <div><span>Data mode</span><strong>{runtime.data.data_mode}</strong></div>
              <div><span>Live connection</span><strong>{runtime.data.live_connected ? 'connected' : 'not configured'}</strong></div>
              <div><span>Default artifact</span><strong>{runtime.data.default_artifact.id} {runtime.data.default_artifact.ready ? '' : '(missing)'} {runtime.data.default_artifact.is_synthetic ? '· synthetic' : ''}</strong></div>
              <div><span>Python</span><strong>{runtime.data.python}</strong></div>
              <div><span>Platform</span><strong>{runtime.data.platform}</strong></div>
              {Object.entries(runtime.data.packages).map(([name, version]) => <div key={name}><span>{name}</span><strong>{version ?? 'missing'}</strong></div>)}
              <div><span>API bind</span><strong>{runtime.data.api_bind}</strong></div>
            </div>
          )}
          <p className="prose small">{runtime.data?.live_note ?? 'Live timing is optional and backend-only; see docs/SECURITY.md.'}</p>
        </Panel>
      </div>
    </div>
  );
}
