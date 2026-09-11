# Blind tester's leakage probe, as run against commit 626fa60. Paths were made configurable
# afterwards (RACESHIFT_ROOT, default ./RaceShift) and the invariants A-E now fail the run
# (exit 1) instead of only printing; the probes themselves are unchanged.
import os, sys, warnings
R = os.environ.get('RACESHIFT_ROOT', 'RaceShift')
sys.path.insert(0, f'{R}/src')
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.simplefilter('ignore', InconsistentVersionWarning)
except ImportError:
    pass
import numpy as np, pandas as pd
from raceshift.models.artifact import RaceShiftArtifact
art = RaceShiftArtifact(f'{R}/artifacts/f1_2025h2_ffr-m')
raw = pd.read_parquet(f'{R}/data/imports/f1_2025_season.parquet')
FAILURES = []
def check(ok, label):
    if not ok: FAILURES.append(label)
r24 = raw[raw.round_number>=23].copy()   # Qatar + Abu Dhabi for speed
def fc(df, driver='VER'):
    return art.forecast_last_available(df, driver=driver)
# A. determinism
a = fc(r24); b = fc(r24)
print('A determinism identical:', a['predicted_next_lap_s']==b['predicted_next_lap_s'], a['predicted_next_lap_s']); check(a['predicted_next_lap_s']==b['predicted_next_lap_s'], 'A determinism')
# B. row order shuffle
sh = r24.sample(frac=1.0, random_state=1)
c = fc(sh); print('B shuffled rows identical:', abs(c['predicted_next_lap_s']-a['predicted_next_lap_s'])<1e-9, c['predicted_next_lap_s']); check(abs(c['predicted_next_lap_s']-a['predicted_next_lap_s'])<1e-9, 'B row order')
# C. truncate VER at lap 40 in Abu Dhabi and forecast lap 41; then compare to full-file table row for lap 40 (features must not depend on later laps)
cut = r24[~((r24.round_number==24)&(r24.driver=='VER')&(r24.lap_number>40))]
f40 = fc(cut)
tab = art.inference_table(r24)
row = tab[(tab.round_number==24)&(tab.driver=='VER')&(tab.lap_number==40)]
feats = art.contract['numeric']+art.contract['categorical']
x = np.asarray(art.preprocessor.transform(row[feats]), dtype=np.float32)
p = art.model.predict_with_uncertainty(x)
full40 = float(row.rolling_median_5.iloc[0] + p['prediction'][0])
print(f'C truncated-at-40 forecast {f40["predicted_next_lap_s"]:.4f} vs full-table lap-40 row {full40:.4f} diff {abs(f40["predicted_next_lap_s"]-full40):.2e} (lap41 actual {float(r24[(r24.round_number==24)&(r24.driver=="VER")&(r24.lap_number==41)].lap_time_s.iloc[0]):.3f})')
check(abs(f40['predicted_next_lap_s']-full40) < 1e-6, 'C truncation')
# D. perturb ONLY lap 41+ of VER (future) hugely -> lap-40 forecast must not change
pert = r24.copy(); m = (pert.round_number==24)&(pert.driver=='VER')&(pert.lap_number>40); pert.loc[m,'lap_time_s'] += 30
tab2 = art.inference_table(pert); row2 = tab2[(tab2.round_number==24)&(tab2.driver=='VER')&(tab2.lap_number==40)]
x2 = np.asarray(art.preprocessor.transform(row2[feats]), dtype=np.float32); p2 = art.model.predict_with_uncertainty(x2)
print('D future-lap perturbation changes lap-40 forecast by', abs(float(row2.rolling_median_5.iloc[0]+p2['prediction'][0]) - full40)); check(abs(float(row2.rolling_median_5.iloc[0]+p2['prediction'][0]) - full40) < 1e-6, 'D future laps')
# E. perturb other drivers' laps in the same race -> should not matter (no gap features used)
pert = r24.copy(); m = (pert.round_number==24)&(pert.driver!='VER'); pert.loc[m,'lap_time_s'] += 5
tab3 = art.inference_table(pert); row3 = tab3[(tab3.round_number==24)&(tab3.driver=='VER')&(tab3.lap_number==40)]
x3 = np.asarray(art.preprocessor.transform(row3[feats]), dtype=np.float32); p3 = art.model.predict_with_uncertainty(x3)
print('E other-driver perturbation changes lap-40 forecast by', abs(float(row3.rolling_median_5.iloc[0]+p3['prediction'][0]) - full40)); check(abs(float(row3.rolling_median_5.iloc[0]+p3['prediction'][0]) - full40) < 1e-6, 'E other drivers')
# F. perturb the EARLIER event (Qatar) -> priors change -> forecast may legitimately change; measure magnitude
pert = r24.copy(); m = (pert.round_number==23); pert.loc[m,'lap_time_s'] += 5
f = fc(pert); print('F earlier-event +5s perturbation changes forecast by', abs(f['predicted_next_lap_s']-a['predicted_next_lap_s']), '(priors: driver_overall before/after', a['historical_context']['driver_overall_pace_s'], f['historical_context']['driver_overall_pace_s'], ')')
# G. event_date vs round_number disagreement: swap event_date so Abu Dhabi looks earlier than Qatar
pert = r24.copy(); pert.loc[pert.round_number==24,'event_date']='2025-01-01'
f = fc(pert); print('G event_date contradicts round_number: forecast', f['predicted_next_lap_s'], 'priors driver_overall', f['historical_context']['driver_overall_pace_s'], '| latest session picked:', f['event'])
# H. season as float / string
pert = r24.copy(); pert['season'] = pert['season'].astype(float)
try: f = fc(pert); print('H float season ok', f['season'], f['predicted_next_lap_s'])
except Exception as e: print('H float season ERROR', type(e).__name__, e)
pert = r24.copy(); pert['season'] = pert['season'].astype(str)
try: f = fc(pert); print('H str season ok', f['season'], f['predicted_next_lap_s'])
except Exception as e: print('H str season ERROR', type(e).__name__, e)
# I. is_accurate all False (a file without the flag would default True; but explicit False everywhere)
pert = r24.copy(); pert['is_accurate']=False
try: f = fc(pert); print('I all inaccurate ok?', f['predicted_next_lap_s'])
except Exception as e: print('I all inaccurate ->', type(e).__name__, str(e)[:120])
# J. track_status as integer dtype
pert = r24.copy(); pert['track_status'] = pd.to_numeric(pert['track_status'])
try: f = fc(pert); print('J int track_status ok', f['predicted_next_lap_s'], 'clean-laps', f['context']['laps_in_segment'])
except Exception as e: print('J int track_status ERROR', type(e).__name__, e)
# K. track_status NaN
pert = r24.copy(); pert['track_status'] = np.nan
try: f = fc(pert); print('K NaN track_status ok', f['predicted_next_lap_s'], 'clean-laps', f['context']['laps_in_segment'])
except Exception as e: print('K NaN track_status ERROR', type(e).__name__, e)
# L. extreme weather values
pert = r24.copy(); pert.loc[pert.driver=='VER',['track_temp_c','air_temp_c','humidity_pct','wind_speed_ms','pressure_mbar']] = [999, -80, 500, 200, 0]
f = fc(pert); print('L absurd weather forecast', f['predicted_next_lap_s'], 'vs normal', a['predicted_next_lap_s'], 'interval', f['lower_80_s'], f['upper_80_s'])
# M. tyre_life 500, stint 40, position 0
pert = r24.copy(); pert.loc[pert.driver=='VER',['tyre_life','stint','position']] = [500, 40, 0]
f = fc(pert); print('M absurd tyre/stint/position forecast', f['predicted_next_lap_s'])
# N. unseen compound / team / driver names
pert = r24.copy(); pert.loc[pert.driver=='VER',['compound','team']] = ['C6', 'Cadillac']
f = fc(pert); print('N unseen compound+team forecast', f['predicted_next_lap_s'], 'team prior', f['historical_context']['team_overall_pace_s'])
# O. slow progressive: last 5 laps each +3s (tyre falling off) -> what does model forecast vs prev lap
pert = r24.copy(); m=(pert.round_number==24)&(pert.driver=='VER')&(pert.lap_number>53); pert.loc[m,'lap_time_s'] += np.arange(1, m.sum()+1)*3.0
f = fc(pert); hist = pert[m].lap_time_s.tolist()
print('O degrading laps', [round(v,2) for v in hist], '-> forecast', round(f['predicted_next_lap_s'],3), 'roll5', round(f['rolling5_baseline_s'],3), 'last', round(f['last_lap_time_s'],3))
# P. Constant lap times
pert = r24.copy(); m=(pert.round_number==24)&(pert.driver=='VER'); pert.loc[m,'lap_time_s']=90.0
f = fc(pert); print('P all laps exactly 90.000 -> forecast', f['predicted_next_lap_s'], 'interval +-', (f['upper_80_s']-f['lower_80_s'])/2, 'disagreement', f['layer_disagreement_s'])

if FAILURES:
    print('LEAKAGE CHECK FAILED:', FAILURES)
    sys.exit(1)
print('leakage invariants A-E hold')
