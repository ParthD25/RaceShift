import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'RaceShift/src')
import pandas as pd, numpy as np
from raceshift.models.artifact import RaceShiftArtifact
from raceshift.features.full_context import lap_state_flags
art = RaceShiftArtifact('RaceShift/artifacts/f1_2025h2_ffr-m')
r25 = pd.read_parquet('RaceShift/data/imports/f1_2025_season.parquet'); r26 = pd.read_parquet('data2026/f1_2026_races.parquet')
comb = pd.concat([r25, r26], ignore_index=True)
table = art.inference_table(comb)     # causal features (verified identical to truncated-file forecasts)
feats = art.contract['numeric'] + art.contract['categorical']
flags = pd.concat([r26, lap_state_flags(r26)], axis=1)
nxt = flags.sort_values(['event','driver','lap_number']).copy()
nxt['next_time'] = nxt.groupby(['event','driver']).lap_time_s.shift(-1); nxt['next_valid'] = nxt.groupby(['event','driver']).lap_valid.shift(-1); nxt['next_lap'] = nxt.groupby(['event','driver']).lap_number.shift(-1)
nxt['next_pit_in'] = nxt.groupby(['event','driver']).is_pit_in.shift(-1); nxt['next_sc'] = (nxt.groupby(['event','driver']).is_safety_car.shift(-1).fillna(False) | nxt.groupby(['event','driver']).is_vsc.shift(-1).fillna(False))
cuts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
rows = []; refused = 0; total = 0
t26 = table[table.season==2026]
for (ev, drv), g in nxt.groupby(['event','driver']):
    for cut in cuts:
        raw_row = g[g.lap_number==cut]
        if raw_row.empty or pd.isna(raw_row.lap_time_s.iloc[0]): continue   # driver not in race / untimed lap: nothing to forecast from
        total += 1
        trow = t26[(t26.event==ev)&(t26.driver==drv)&(t26.lap_number==cut)]
        if trow.empty: refused += 1; continue                                  # product refuses: lap N itself invalid
        x = np.asarray(art.preprocessor.transform(trow[feats]), dtype=np.float32); p = art.model.predict_with_uncertainty(x)
        base = float(trow.rolling_median_5.iloc[0]); pred = base + float(p['prediction'][0])
        rr = raw_row.iloc[0]
        rows.append({'event':ev,'driver':drv,'cut':cut,'pred':pred,'lo':base+float(p['lower_80'][0]),'hi':base+float(p['upper_80'][0]),'prev':float(rr.lap_time_s),'actual':rr.next_time,'next_valid':bool(rr.next_valid) if pd.notna(rr.next_valid) else None,'next_adjacent':rr.next_lap==cut+1,'next_pit_in':bool(rr.next_pit_in) if pd.notna(rr.next_pit_in) else None,'next_sc':bool(rr.next_sc)})
d = pd.DataFrame(rows); d = d[d.actual.notna() & d.next_adjacent]
d['ae'] = (d.pred-d.actual).abs(); d['ae_prev'] = (d.prev-d.actual).abs(); d['inside'] = (d.actual>=d.lo)&(d.actual<=d.hi)
print(f'live-style forecasts issued: {len(rows)} of {total} driver/cut points ({refused} refused because lap N was pit/SC/inaccurate); {len(d)} had a timed adjacent next lap')
def rep(name, q): print(f'  {name:55s} n={len(q):5d} FFR MAE={q.ae.mean():.3f} median={q.ae.median():.3f} p90={q.ae.quantile(.9):.3f} <0.5s={(q.ae<=0.5).mean():.3f} cov80={q.inside.mean():.3f} | prev-lap MAE={q.ae_prev.mean():.3f} <0.5s={(q.ae_prev<=0.5).mean():.3f}')
rep('ALL next laps (what a live user would see)', d)
rep('next lap was a clean racing lap (scored protocol)', d[d.next_valid==True])
rep('next lap NOT clean (pit-in / SC / VSC / inaccurate)', d[d.next_valid!=True])
rep('  of which next lap = pit-in', d[d.next_pit_in==True])
rep('  of which next lap under SC/VSC', d[d.next_sc==True])
print('share of live forecasts whose next lap was not clean:', round(float((d.next_valid!=True).mean()),3))
print('per-cut (all next laps):'); print(d.groupby('cut').agg(n=('ae','size'), FFR=('ae','mean'), prev=('ae_prev','mean'), cov=('inside','mean')).round(3).to_string())
print('per-cut (clean next laps only):'); print(d[d.next_valid==True].groupby('cut').agg(n=('ae','size'), FFR=('ae','mean'), prev=('ae_prev','mean'), cov=('inside','mean')).round(3).to_string())
