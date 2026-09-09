// Typed client for the RaceShift local API. All paths are relative so the Vite proxy
// (dev) or same-origin hosting decides where the FastAPI backend lives.

export type Metrics = {
  mae_s: number;
  rmse_s: number;
  median_ae_s?: number;
  p90_ae_s?: number;
  signed_bias_s?: number;
  rows?: number;
  interval80_coverage?: number;
  interval80_width_s?: number;
};

export type Split = { train_end: number; validation: number; test: number } | null;

export type HealthResponse = {
  status: string;
  version: string;
  local_first: boolean;
  training_policy: string;
  demo_artifact_ready: boolean;
};

export type RuntimeResponse = {
  raceshift_version: string;
  python: string;
  platform: string;
  packages: Record<string, string | null>;
  data_mode: string;
  live_connected: boolean;
  live_note: string;
  default_artifact: { id: string; ready: boolean; is_synthetic: boolean; data_source: string };
  project_root: string;
  api_bind: string;
  ui_origins: string[];
};

export type SetupCheck = { item: string; ok: boolean; detail: string };
export type SetupResponse = { checks: SetupCheck[]; all_required_ok: boolean };

export type ArchitectureLayer = { layer: number; input_nodes: number; hidden_nodes: number; ordinal_groups: number };

export type ArtifactEntry = {
  id: string;
  name: string;
  role: string;
  training: string | null;
  path: string;
  ready: boolean;
  missing_files: string[];
  is_default: boolean;
  is_synthetic: boolean;
  data_source: string;
  architecture: ArchitectureLayer[] | null;
  split: Split;
  validation: Metrics | null;
  test: Metrics | null;
  created_utc: string | null;
};

export type BaselineEntry = { id: string; name: string; role: string; training: string; ready: boolean };
export type ModelsResponse = { default_artifact: string; artifacts: ArtifactEntry[]; baselines: BaselineEntry[] };

export type DatasetSource = { name: string; url: string; status?: 'used' | 'planned'; role: string; coverage_note: string; local_policy: string };
export type ImportFile = { name: string; bytes: number; modified_utc: string };
export type DatasetsResponse = { sources: DatasetSource[]; imports: ImportFile[]; processed_files: string[]; import_dir: string };

export type ImportSummary = {
  file: string;
  rows: number;
  columns: string[];
  missing_required_columns: string[];
  is_synthetic: boolean;
  data_source?: string;
  seasons?: number[];
  events?: string[];
  drivers?: string[];
  latest_session?: { season: number; event: string; session: string };
  latest_session_drivers?: string[];
  bytes?: number;
  stored_as?: string;
};

export type ForecastRequest = { file: string; driver?: string | null; artifact?: string | null };

export type ForecastContext = {
  compound: string | null;
  tyre_life: number | null;
  stint: number | null;
  position: number | null;
  laps_in_segment: number | null;
  track_temp_c: number | null;
  air_temp_c: number | null;
  humidity_pct: number | null;
  wind_speed_ms: number | null;
  wind_direction_deg: number | null;
  rainfall: string | null;
  gap_ahead_s: number | null;
  gap_behind_s: number | null;
  team: string | null;
  circuit: string | null;
};

export type HistoricalContext = {
  driver_circuit_pace_s: number | null;
  team_circuit_pace_s: number | null;
  compound_circuit_pace_s: number | null;
  driver_circuit_compound_pace_s: number | null;
  matched_weather_compound_pace_s: number | null;
  driver_matched_weather_pace_s: number | null;
  driver_overall_pace_s: number | null;
  team_overall_pace_s: number | null;
  note: string;
};

export type ForecastResult = {
  context: ForecastContext;
  historical_context: HistoricalContext;
  season: number;
  event: string;
  session: string;
  driver: string;
  available_drivers: string[];
  lap_number_completed: number;
  completed_laps_in_session: number;
  history_laps_used: number;
  short_history: boolean;
  predicted_next_lap_s: number;
  lower_80_s: number;
  upper_80_s: number;
  layer_disagreement_s: number;
  rolling5_baseline_s: number;
  last_lap_time_s: number;
  artifact: string;
  data_source: string;
  is_synthetic: boolean;
  file: string;
};

export type BacktestRequest = ForecastRequest & { laps?: number };

export type BacktestLap = {
  lap_number_completed: number;
  next_lap_number: number;
  actual_next_lap_s: number;
  predicted_next_lap_s: number;
  lower_80_s: number;
  upper_80_s: number;
  rolling5_baseline_s: number;
  previous_lap_s: number;
  error_s: number;
  abs_error_s: number;
  within_interval: boolean;
  compound: string | null;
  tyre_life: number | null;
  position: number | null;
};

export type BacktestSummary = {
  rows: number;
  mae_s: number;
  rmse_s: number;
  p90_ae_s: number;
  signed_bias_s: number;
  within_0_5s_share: number;
  within_1s_share: number;
  interval80_coverage: number;
  previous_lap_mae_s: number;
  rolling5_mae_s: number;
};

export type BacktestResult = {
  season: number;
  event: string;
  session: string;
  driver: string;
  available_drivers: string[];
  laps_driven: number;
  usable_lap_pairs: number;
  skipped_laps: number;
  laps: BacktestLap[];
  summary: BacktestSummary;
  artifact: string;
  data_source: string;
  is_synthetic: boolean;
  file: string;
};

export type BreakdownRow = { rows: number } & Record<string, number>;
export type ReportEntry = {
  name: string;
  title: string | null;
  models: string[];
  rows: number | null;
  breakdowns: Record<string, Record<string, BreakdownRow>> | null;
  has_summary: boolean;
};
export type ReportsResponse = { reports: ReportEntry[] };

export type DriverMetrics = Record<string, Metrics>;
export type DriverReportEntry = { artifact: string; name: string; is_synthetic: boolean; split: Split; test_by_driver: DriverMetrics };
export type DriversReportResponse = { artifacts: DriverReportEntry[] };

export type RunResources = {
  training?: { wall_seconds?: number; peak_rss_mb?: number; peak_traced_mb?: number };
  inference_batch_ms_per_row?: number;
  inference_single_row_ms?: number;
  artifact_bytes?: number;
};

export type ExperimentRun = {
  run: string;
  model: string;
  method: string | null;
  training_policy: string | null;
  architecture?: ArchitectureLayer[] | null;
  validation: Metrics | null;
  test: Metrics | null;
  resources?: RunResources | null;
  artifact: string;
  data_source: string;
  is_synthetic: boolean;
  split: Split;
  created_utc: string | null;
  input_file: string | null;
};
export type ExperimentsResponse = { experiments: ExperimentRun[] };

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function parse<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;
  let detail = response.statusText;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === 'string') detail = body.detail;
    else if (body.detail) detail = JSON.stringify(body.detail);
  } catch {
    // keep statusText
  }
  throw new ApiError(response.status, detail || `Request failed (${response.status})`);
}

const getJson = <T,>(url: string) => fetch(url).then(parse<T>);

export const api = {
  health: () => getJson<HealthResponse>('/api/health'),
  runtime: () => getJson<RuntimeResponse>('/api/runtime'),
  setup: () => getJson<SetupResponse>('/api/setup'),
  models: () => getJson<ModelsResponse>('/api/models'),
  modelExportUrl: (id: string) => `/api/models/${encodeURIComponent(id)}/export`,
  datasets: () => getJson<DatasetsResponse>('/api/datasets'),
  experiments: () => getJson<ExperimentsResponse>('/api/experiments'),
  importSummary: (file: string) => getJson<ImportSummary>(`/api/imports/${encodeURIComponent(file)}/summary`),
  forecastLatest: (body: ForecastRequest) =>
    fetch('/api/forecast/latest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(parse<ForecastResult>),
  backtest: (body: BacktestRequest) =>
    fetch('/api/forecast/backtest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(parse<BacktestResult>),
  reports: () => getJson<ReportsResponse>('/api/reports'),
  driverReports: () => getJson<DriversReportResponse>('/api/reports/drivers'),
  importFile: (file: File, overwrite: boolean) => {
    const form = new FormData();
    form.append('file', file, file.name);
    return fetch(`/api/import?overwrite=${overwrite ? 'true' : 'false'}`, { method: 'POST', body: form }).then(parse<ImportSummary>);
  }
};

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof TypeError) return 'Local API is not reachable. Start it with `npm run dev`.';
  if (error instanceof Error) return error.message;
  return String(error);
}

export function fmtLap(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—';
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(3).padStart(6, '0')}`;
}

export function fmtDelta(seconds: number | null | undefined, digits = 3): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—';
  return `${seconds > 0 ? '+' : ''}${seconds.toFixed(digits)}s`;
}

export function fmtNumber(value: number | null | undefined, digits = 3): string {
  return value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
}

export function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function architectureLabel(layers: ArchitectureLayer[] | null | undefined): string {
  if (!layers || layers.length === 0) return '—';
  return [layers[0].input_nodes, ...layers.map(l => l.hidden_nodes)].join(' → ');
}
