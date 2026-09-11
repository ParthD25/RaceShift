export type SourceKind = 'fixture' | 'local' | 'synthetic' | 'real' | 'offline';

const LABELS: Record<SourceKind, string> = {
  fixture: 'Fixture visual',
  local: 'Local data',
  synthetic: 'Synthetic model',
  real: 'Real artifact',
  offline: 'API offline'
};

const TITLES: Record<SourceKind, string> = {
  fixture: 'Illustrative placeholder content. Not produced by a model or by real Formula 1 data.',
  local: 'Read from a file in data/imports through the local API.',
  synthetic: 'Model trained on the synthetic engineering fixture. Its metrics are not Formula 1 results.',
  real: 'Real Formula 1 timing data (FastF1), or a model trained on it.',
  offline: 'The local FastAPI backend did not respond.'
};

export function SourceBadge({ kind, text }: { kind: SourceKind; text?: string }) {
  return (
    <span className={`badge badge-${kind}`} title={TITLES[kind]}>
      <i />
      {text ?? LABELS[kind]}
    </span>
  );
}

export function sourceKindFor(isSynthetic: boolean | undefined): SourceKind {
  return isSynthetic ? 'synthetic' : 'real';
}
