import { FlaskConical, RefreshCw } from 'lucide-react';
import { Panel } from '../components/Panel';
import { SourceBadge, sourceKindFor } from '../components/SourceBadge';
import { api, fmtNumber } from '../lib/api';
import { useApi } from '../lib/useApi';

export default function Experiments() {
  const experiments = useApi(() => api.experiments());
  const runs = experiments.data?.experiments ?? [];
  return (
    <div className="page-stack">
      <div className="page-heading">
        <div><h1>Experiments</h1><p>Every metrics.json under artifacts/, including baseline reports. Synthetic runs are labelled and are never Formula 1 results.</p></div>
        <button className="primary-btn" onClick={experiments.reload}><RefreshCw size={15} /> Refresh</button>
      </div>
      <Panel title="Experiment registry" icon={<FlaskConical size={17} />} action={experiments.error ? <SourceBadge kind="offline" /> : <SourceBadge kind="local" text="Read from artifacts/" />}>
        {experiments.error && <div className="error-box">{experiments.error}</div>}
        {!experiments.error && !experiments.loading && runs.length === 0 && (
          <div className="empty-inline">No runs found. Train with <code>scripts/train_ffr.py</code> or <code>scripts/train_baselines.py</code> and copy the output folder into <code>artifacts/</code>.</div>
        )}
        {runs.length > 0 && (
          <div className="wide-table experiments-table">
            <div className="wide-head"><span>Run</span><span>Method</span><span>Split</span><span>Val MAE</span><span>Test MAE</span><span>Test RMSE</span><span>80% cov.</span><span>Train s</span><span>Peak MB</span><span>Source</span></div>
            {runs.map(r => (
              <div className="wide-row" key={r.run}>
                <span><strong>{r.run}</strong></span>
                <span>{r.method ?? '—'}</span>
                <span>{r.split ? `≤${r.split.train_end} / ${r.split.validation} / ${r.split.test}` : '—'}</span>
                <span>{fmtNumber(r.validation?.mae_s)}</span>
                <span>{fmtNumber(r.test?.mae_s)}</span>
                <span>{fmtNumber(r.test?.rmse_s)}</span>
                <span>{r.test?.interval80_coverage != null ? `${(r.test.interval80_coverage * 100).toFixed(1)}%` : '—'}</span>
                <span>{r.resources?.training?.wall_seconds != null && r.resources.training.wall_seconds > 0 ? fmtNumber(r.resources.training.wall_seconds, 1) : '—'}</span>
                <span>{r.resources?.training?.peak_rss_mb ? fmtNumber(r.resources.training.peak_rss_mb, 0) : '—'}</span>
                <span><SourceBadge kind={sourceKindFor(r.is_synthetic)} text={r.is_synthetic ? 'Synthetic' : 'Real data'} /></span>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}
