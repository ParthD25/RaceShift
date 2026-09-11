import sys, warnings, json; warnings.filterwarnings('ignore')
sys.path.insert(0, 'RaceShift/src')
import numpy as np, pandas as pd
from raceshift.models.artifact import RaceShiftArtifact
from raceshift.train.metrics import regression_metrics, interval_metrics
from raceshift.features.full_context import RAW_TARGET_COLUMN
pd.set_option('display.width', 250)
r25 = pd.read_parquet('RaceShift/data/imports/f1_2025_season.parquet')
r26 = pd.read_parquet('data2026/f1_2026_races.parquet')
combined = pd.concat([r25, r26], ignore_index=True)
combined.to_parquet('RaceShift/data/imports/f1_2025_2026.parquet', index=False)
UNSEEN_DRIVERS = {'ANT','BOR','HAD','LIN'}; UNSEEN_TEAMS = {'Audi','Cadillac'}
def score(art, raw, label):
    tab = art.inference_table(raw)
    feats = art.contract['numeric'] + art.contract['categorical']
    x = np.asarray(art.preprocessor.transform(tab[feats]), dtype=np.float32)
    p = art.model.predict_with_uncertainty(x); base = tab['rolling_median_5'].to_numpy(float)
    tab['pred'] = base + p['prediction']; tab['lo'] = base + p['lower_80']; tab['hi'] = base + p['upper_80']; tab['disagree'] = p['layer_disagreement']
    s = tab[(tab.season==2026) & tab[RAW_TARGET_COLUMN].notna()].copy()
    s['err'] = s['pred'] - s[RAW_TARGET_COLUMN]; s['ae'] = s['err'].abs(); s['ae_prev'] = (s['lap_time_s'] - s[RAW_TARGET_COLUMN]).abs(); s['ae_r5'] = (s['rolling_median_5'] - s[RAW_TARGET_COLUMN]).abs()
    m = regression_metrics(s[RAW_TARGET_COLUMN], s['pred']); m.update(interval_metrics(s[RAW_TARGET_COLUMN], s['lo'], s['hi']))
    pl = regression_metrics(s[RAW_TARGET_COLUMN], s['lap_time_s']); r5 = regression_metrics(s[RAW_TARGET_COLUMN], s['rolling_median_5'])
    print(f"\n### {label}: 2026 scored rows={m['rows']} | FFR MAE={m['mae_s']:.4f} RMSE={m['rmse_s']:.3f} med={m['median_ae_s']:.3f} p90={m['p90_ae_s']:.3f} <0.5={m['within_0_5s_share']:.3f} <1={m['within_1s_share']:.3f} cov80={m['interval80_coverage']:.3f} width={m['interval80_width_s']:.3f} bias={m['signed_bias_s']:+.3f} | prev_lap MAE={pl['mae_s']:.4f} RMSE={pl['rmse_s']:.3f} <0.5={pl['within_0_5s_share']:.3f} | roll5 MAE={r5['mae_s']:.4f}")
    return s
def brk(s, by, label, minn=20):
    rows=[]
    for k,g in s.groupby(by, dropna=False):
        if len(g)<minn: continue
        rows.append({by:k,'n':len(g),'FFR':g.ae.mean(),'prev':g.ae_prev.mean(),'roll5':g.ae_r5.mean(),'FFR_p90':g.ae.quantile(.9),'cov80':((g[RAW_TARGET_COLUMN]>=g.lo)&(g[RAW_TARGET_COLUMN]<=g.hi)).mean(),'bias':g.err.mean()})
    d = pd.DataFrame(rows).sort_values('n', ascending=False); print(f'-- by {label}'); print(d.round(3).to_string(index=False))
ffrm = RaceShiftArtifact('RaceShift/artifacts/f1_2025h2_ffr-m')
s = score(ffrm, combined, 'FFR-M, 2025+2026 file (priors from 2025)')
s.to_parquet('eval_2026_ffrm_scored.parquet', index=False)
brk(s, 'event', 'event')
s['team_seen'] = np.where(s.team.isin(UNSEEN_TEAMS), 'UNSEEN team ('+s.team+')', 'seen team'); brk(s, 'team_seen', 'team seen/unseen')
brk(s, 'team', 'team')
s['driver_seen'] = np.where(s.driver.isin(UNSEEN_DRIVERS), 'UNSEEN driver', 'seen driver'); brk(s, 'driver_seen', 'driver seen/unseen')
brk(s, 'driver', 'driver')
brk(s, 'compound', 'compound')
brk(s, 'rainfall', 'rainfall flag')
s['seg'] = pd.cut(s.laps_in_segment, [0,1,4,10,1000], labels=['1','2-4','5-10','11+']); brk(s, 'seg', 'consecutive clean laps')
s['pos'] = pd.cut(s.position, [0,3,10,30], labels=['P1-3','P4-10','P11+']); brk(s, 'pos', 'position')
s['tyre'] = pd.cut(s.tyre_life, [0,5,15,30,100], labels=['1-5','6-15','16-30','31+']); brk(s, 'tyre', 'tyre age')
s['next_yellow'] = s.groupby(['event','driver'])['is_yellow'].shift(-1); brk(s, 'is_yellow', 'yellow flag on lap N')
print('-- worst 15 errors:'); print(s.sort_values('ae', ascending=False)[['event','driver','lap_number','lap_time_s','rolling_median_5','pred',RAW_TARGET_COLUMN,'err','laps_in_segment','compound','tyre_life','track_status','is_yellow']].head(15).round(3).to_string(index=False))
print('-- error distribution:', s.ae.describe(percentiles=[.5,.9,.95,.99]).round(3).to_dict())
print('-- share of laps where FFR beats previous_lap:', float((s.ae < s.ae_prev).mean()), ' ties within 0.01:', float(((s.ae-s.ae_prev).abs()<0.01).mean()))
print('-- interval half-width stats:', ((s.hi-s.lo)/2).describe().round(3).to_dict(), ' share widened by disagreement:', float((s.disagree > ffrm.model.interval_q80).mean()))
# 2026-only file (no priors at all)
s2 = score(ffrm, r26, 'FFR-M, 2026-only file (no priors)')
mg = s.merge(s2[['event','driver','lap_number','pred']], on=['event','driver','lap_number'], suffixes=('','_noprior'))
print('  pred diff with vs without 2025 priors: mean abs', float((mg.pred-mg.pred_noprior).abs().mean()), 'max', float((mg.pred-mg.pred_noprior).abs().max()))
# legacy FFR-S
leg = RaceShiftArtifact('RaceShift/artifacts/f1_2025h2_legacy_ext_ffr-s')
s3 = score(leg, combined, 'legacy FFR-S (2000-2024 train), 2025+2026 file')
brk(s3, 'event', 'event (legacy FFR-S)')
# demo synthetic on 2026 (what the UI lets you do)
demo = RaceShiftArtifact('RaceShift/artifacts/raceshift_ffr_demo')
s4 = score(demo, combined, 'SYNTHETIC demo artifact on 2026 real laps')
# Sprint + Q sessions
for f in ['data2026/2026_Chinese_Grand_Prix_S.parquet','data2026/2026_Dutch_Grand_Prix_S.parquet','data2026/2026_Italian_Grand_Prix_Q.parquet']:
    d = pd.read_parquet(f)
    try:
        ss = score(ffrm, pd.concat([r25, d], ignore_index=True), f'FFR-M on {f.split("/")[-1]}')
        if len(ss): print('   worst 3:', ss.sort_values('ae', ascending=False)[['driver','lap_number','lap_time_s','pred',RAW_TARGET_COLUMN,'err','laps_in_segment']].head(3).round(3).to_dict('records'))
    except Exception as e: print(f, 'ERROR', type(e).__name__, e)
