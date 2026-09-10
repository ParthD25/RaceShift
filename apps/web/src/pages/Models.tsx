import { BrainCircuit, Cpu } from 'lucide-react';
import { Panel } from '../components/Panel';
import { SourceBadge, sourceKindFor } from '../components/SourceBadge';
import { api, architectureLabel, fmtNumber } from '../lib/api';
import { useApi } from '../lib/useApi';

export default function Models() {
  const models = useApi(() => api.models());
  const artifacts = models.data?.artifacts ?? [];
  const baselines = models.data?.baselines ?? [];
  return (
    <div className="page-stack">
      <div className="page-heading"><div><h1>Models</h1><p>Saved artifacts in artifacts/ and the baselines every artifact must beat.</p></div></div>
      {models.error && <div className="error-box">{models.error}</div>}
      <div className="model-grid">
        {artifacts.map(a => (
          <Panel key={a.id} title={a.label ?? a.id} icon={<Cpu size={17} />} action={<SourceBadge kind={a.ready ? sourceKindFor(a.is_synthetic) : 'offline'} text={a.role === 'baseline-report' ? 'Metrics only' : a.ready ? undefined : 'Incomplete'} />}>
            <div className="model-role">{a.role === 'baseline-report' ? 'Baseline report' : a.name}{a.is_default ? ' · default' : ''}</div>
            <div className="detail-list compact">
              <div><span>Folder</span><strong>{a.id}</strong></div>
              {a.role === 'baseline-report'
                ? <div><span>Contents</span><strong>previous lap, rolling median, ridge and tree baselines scored on this split; no weights to load, see Experiments</strong></div>
                : <>
                    <div><span>Nodes</span><strong>{architectureLabel(a.architecture)}</strong></div>
                    <div><span>Ordinal groups</span><strong>{a.architecture?.map(l => l.ordinal_groups).join(' / ') ?? '—'}</strong></div>
                    <div><span>Test MAE</span><strong>{fmtNumber(a.test?.mae_s)} s</strong></div>
                  </>}
              <div><span>Data</span><strong>{a.data_source}</strong></div>
              {!a.ready && a.role !== 'baseline-report' && <div><span>Missing</span><strong className="bad">{a.missing_files.join(', ')}</strong></div>}
              {a.ready && a.role !== 'baseline-report' && <div><span>Export</span><strong><a href={api.modelExportUrl(a.id)} download>{a.id}.zip</a> · ONNX core, JSON preprocessor, model card</strong></div>}
            </div>
          </Panel>
        ))}
        {baselines.map(b => (
          <Panel key={b.id} title={b.name} icon={<Cpu size={17} />}>
            <div className="model-role">{b.role.replace(/-/g, ' ')}</div>
            <div className="detail-list compact">
              <div><span>Training</span><strong>{b.training}</strong></div>
              <div><span>Computed by</span><strong>scripts/train_baselines.py</strong></div>
            </div>
          </Panel>
        ))}
      </div>
      <Panel title="Selection rule" icon={<BrainCircuit size={17} />}>
        <p className="prose">RaceShift FFR is the research focus, not a presumption. It enters the production path only if it beats the previous-lap, rolling-five, ridge and tree baselines on an unseen future season, keeps its 80% interval calibrated, and trains inside the Colab budget with no global backpropagation.</p>
      </Panel>
    </div>
  );
}
