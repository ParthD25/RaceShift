import sys, time, json
sys.path.insert(0, 'RaceShift/src')
import numpy as np, pandas as pd
from raceshift.models.artifact import RaceShiftArtifact
from raceshift.train.metrics import regression_metrics, interval_metrics
from raceshift.features.full_context import RAW_TARGET_COLUMN

art = RaceShiftArtifact('RaceShift/artifacts/f1_2025h2_ffr-m')
raw = pd.read_parquet('RaceShift/data/imports/f1_2025_season.parquet')
t = time.time()
table = art.inference_table(raw)
print(f'inference table: {len(table)} rows in {time.time()-t:.1f}s; raw rows {len(raw)}')
feats = art.contract['numeric'] + art.contract['categorical']
x = np.asarray(art.preprocessor.transform(table[feats]), dtype=np.float32)
print('encoded dim', x.shape)
pred = art.model.predict_with_uncertainty(x)
base = table['rolling_median_5'].to_numpy(float)
table['pred'] = base + pred['prediction']; table['lo'] = base + pred['lower_80']; table['hi'] = base + pred['upper_80']
scored = table[table[RAW_TARGET_COLUMN].notna()].copy()
test = scored[scored.round_number > 12]
val = scored[scored.round_number <= 12]
def rep(name, d):
    m = regression_metrics(d[RAW_TARGET_COLUMN], d['pred']); m.update(interval_metrics(d[RAW_TARGET_COLUMN], d['lo'], d['hi']))
    pl = regression_metrics(d[RAW_TARGET_COLUMN], d['lap_time_s']); r5 = regression_metrics(d[RAW_TARGET_COLUMN], d['rolling_median_5'])
    print(f"{name}: rows={m['rows']} FFR-M MAE={m['mae_s']:.4f} RMSE={m['rmse_s']:.3f} p90={m['p90_ae_s']:.3f} <0.5s={m['within_0_5s_share']:.3f} <1s={m['within_1s_share']:.3f} cov80={m['interval80_coverage']:.3f} bias={m['signed_bias_s']:+.3f} | prev_lap MAE={pl['mae_s']:.4f} | roll5 MAE={r5['mae_s']:.4f}")
    return m
rep('2025 test rounds 13-24 (single-season file, priors from 2025 only)', test)
rep('2025 val rounds 1-12', val)
# Compare with committed test_predictions.csv (trained with 2018-2024 priors)
comm = pd.read_csv('RaceShift/artifacts/f1_2025h2_ffr-m/test_predictions.csv')
cm = regression_metrics(comm.actual_next_lap_s, comm.predicted_next_lap_s); cm.update(interval_metrics(comm.actual_next_lap_s, comm.lower_80_s, comm.upper_80_s))
print('committed csv metrics recomputed:', {k: round(v,4) for k,v in cm.items()})
merged = comm.merge(test[['season','round_number','event','driver','lap_number', RAW_TARGET_COLUMN, 'pred']], on=['season','round_number','event','driver','lap_number'], how='outer', indicator=True)
print('row match:', merged._merge.value_counts().to_dict())
both = merged[merged._merge=='both']
print('actual mismatch max:', float((both.actual_next_lap_s - both[RAW_TARGET_COLUMN]).abs().max()))
d = (both.predicted_next_lap_s - both.pred)
print(f'prediction diff (committed vs re-run on shipped file): mean abs {d.abs().mean():.4f}s, max {d.abs().max():.3f}s, share > 0.05s {(d.abs()>0.05).mean():.3f}, share >0.2s {(d.abs()>0.2).mean():.3f}')
# Per-event on the test half
for ev, g in test.groupby('event'):
    m = regression_metrics(g[RAW_TARGET_COLUMN], g['pred']); pl = regression_metrics(g[RAW_TARGET_COLUMN], g['lap_time_s'])
    print(f'  {ev:28s} n={m["rows"]:5d} FFR-M {m["mae_s"]:.3f}  prev_lap {pl["mae_s"]:.3f}')
# prior availability
print('hist prior coverage in test rows:', {c: round(float(test[c].notna().mean()),3) for c in ['hist_driver_circuit_pace','hist_team_circuit_pace','hist_driver_global_pace','hist_weather_compound_pace']})
