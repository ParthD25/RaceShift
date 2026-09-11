import json, time, io, sys, os
import requests, pandas as pd, numpy as np
B = 'http://127.0.0.1:8000'
IMP = 'RaceShift/data/imports'
def call(method, path, label=None, **kw):
    t = time.time()
    r = requests.request(method, B + path, timeout=300, **kw)
    dt = time.time() - t
    body = r.text
    try: body = json.dumps(r.json())[:260]
    except Exception: body = body[:200]
    print(f'[{r.status_code}] {dt:6.2f}s {label or path}: {body}')
    return r
print('== baseline forecast timings ==')
call('POST', '/api/forecast/latest', 'first forecast default file', json={'file':'f1_2025_season.parquet'})
call('POST', '/api/forecast/latest', 'second forecast (cached)', json={'file':'f1_2025_season.parquet'})
call('POST', '/api/forecast/backtest', 'backtest VER 10', json={'file':'f1_2025_season.parquet','driver':'VER','laps':10})
print('== driver / param edge cases ==')
call('POST', '/api/forecast/latest', 'unknown driver', json={'file':'f1_2025_season.parquet','driver':'XXX'})
call('POST', '/api/forecast/latest', 'lowercase driver', json={'file':'f1_2025_season.parquet','driver':'ver'})
call('POST', '/api/forecast/latest', 'driver with spaces', json={'file':'f1_2025_season.parquet','driver':' VER '})
call('POST', '/api/forecast/latest', 'driver numeric', json={'file':'f1_2025_season.parquet','driver':1})
call('POST', '/api/forecast/backtest', 'laps=0', json={'file':'f1_2025_season.parquet','laps':0})
call('POST', '/api/forecast/backtest', 'laps=101', json={'file':'f1_2025_season.parquet','laps':101})
call('POST', '/api/forecast/backtest', 'laps=100 (more than driven)', json={'file':'f1_2025_season.parquet','driver':'VER','laps':100})
call('POST', '/api/forecast/backtest', 'laps="abc"', json={'file':'f1_2025_season.parquet','laps':'abc'})
call('POST', '/api/forecast/backtest', 'laps=2.5', json={'file':'f1_2025_season.parquet','laps':2.5})
call('POST', '/api/forecast/latest', 'empty body', data='')
call('POST', '/api/forecast/latest', 'malformed json', data='{bad', headers={'Content-Type':'application/json'})
print('== path handling ==')
for f in ['../../etc/passwd', '..%2F..%2Fetc%2Fpasswd', '/etc/passwd', 'f1_2025_season.parquet/../synthetic_fixture.csv', 'nope.parquet', 'f1_2025_season.PARQUET', '', '.', '..', 'a.txt', 'synthetic_fixture.csv\x00.parquet']:
    call('POST', '/api/forecast/latest', f'file={f!r}', json={'file':f})
call('GET', '/api/forecast?file=..%2F..%2Fpyproject.toml', 'GET alias traversal')
call('GET', '/api/imports/..%2F..%2Fpyproject.toml/summary', 'summary traversal')
call('GET', '/api/imports/%2e%2e%2f%2e%2e%2fpyproject.toml/summary', 'summary traversal 2')
for a in ['../data', '../../', 'nonexistent', 'f1_2025h2_baselines', 'domain_shift_2026_baselines', '', 'raceshift_ffr_demo']:
    call('POST', '/api/forecast/latest', f'artifact={a!r}', json={'file':'f1_2025_season.parquet','artifact':a})
call('GET', '/api/models/..%2F..%2Fdata/export', 'export traversal')
call('GET', '/api/models/f1_2025h2_baselines/export', 'export baseline folder')
print('== other artifacts on real data ==')
call('POST', '/api/forecast/latest', 'synthetic demo model on real laps', json={'file':'f1_2025_season.parquet','artifact':'raceshift_ffr_demo','driver':'VER'})
call('POST', '/api/forecast/backtest', 'synthetic demo backtest on real laps', json={'file':'f1_2025_season.parquet','artifact':'raceshift_ffr_demo','driver':'VER','laps':10})
call('POST', '/api/forecast/backtest', 'legacy FFR-S backtest VER', json={'file':'f1_2025_season.parquet','artifact':'f1_2025h2_legacy_ext_ffr-s','driver':'VER','laps':10})
call('POST', '/api/forecast/latest', 'real model on synthetic fixture', json={'file':'synthetic_fixture.csv'})
print('== export ==')
t=time.time(); r = requests.get(B+'/api/models/f1_2025h2_ffr-m/export', timeout=600); print(f'[{r.status_code}] {time.time()-t:.1f}s export FFR-M: {len(r.content)} bytes, ct={r.headers.get("content-type")}')
r = requests.get(B+'/api/models/raceshift_ffr_demo/export', timeout=600); print(f'[{r.status_code}] export demo: {len(r.content)} bytes')
print('== import uploads ==')
def up(name, content, label, overwrite=False, ctype=None):
    files = {'file': (name, content)}
    return call('POST', f'/api/import?overwrite={"true" if overwrite else "false"}', f'upload {label}', files=files)
up('empty.csv', b'', 'empty file')
up('onecol.csv', b'a\n1\n2\n', 'single column')
up('hdr.csv', b'season,event\n', 'header only')
up('evil.txt', b'season,event\n1,2\n', '.txt extension')
up('../../evil.csv', b'season,event\n1,2\n', 'traversal filename')
up('weird name (1).csv', b'season,event\n2025,x\n', 'weird filename')
up('bad.parquet', b'not a parquet', 'bad parquet bytes')
up('f1_2025_season.parquet', b'season,event\n2025,x\n', 'overwrite shipped file without flag')
# minimal-columns CSV, then forecast on it
raw = pd.read_parquet(f'{IMP}/f1_2025_season.parquet')
mini = raw[raw.round_number==24][['season','event','session','driver','lap_number','lap_time_s']]
up('minimal.csv', mini.to_csv(index=False).encode(), 'minimal required columns only (no chronology col)')
call('POST', '/api/forecast/latest', 'forecast on minimal.csv', json={'file':'minimal.csv','driver':'VER'})
mini2 = raw[raw.round_number==24][['season','round_number','event','session','driver','lap_number','lap_time_s']]
up('minimal2.csv', mini2.to_csv(index=False).encode(), 'minimal + round_number')
call('POST', '/api/forecast/latest', 'forecast on minimal2.csv', json={'file':'minimal2.csv','driver':'VER'})
call('POST', '/api/forecast/backtest', 'backtest on minimal2.csv', json={'file':'minimal2.csv','driver':'VER','laps':10})
# string lap times / garbage
g = mini2.copy(); g['lap_time_s'] = g['lap_time_s'].astype(str); g.loc[g.index[:5], 'lap_time_s'] = 'fast'
up('strings.csv', g.to_csv(index=False).encode(), 'string lap times')
call('POST', '/api/forecast/latest', 'forecast on strings.csv', json={'file':'strings.csv','driver':'VER'})
# negative and huge lap times
g = mini2.copy(); g.loc[g.driver=='VER','lap_time_s'] = g.loc[g.driver=='VER','lap_time_s'].where(g.lap_number%7!=0, -5.0)
up('negative.csv', g.to_csv(index=False).encode(), 'negative lap times every 7th lap')
call('POST', '/api/forecast/backtest', 'backtest negative.csv VER', json={'file':'negative.csv','driver':'VER','laps':10})
g = mini2.copy(); g.loc[(g.driver=='VER')&(g.lap_number==57),'lap_time_s'] = 1e9
up('huge.csv', g.to_csv(index=False).encode(), 'lap 57 = 1e9 s')
call('POST', '/api/forecast/latest', 'forecast huge.csv VER', json={'file':'huge.csv','driver':'VER'})
g = mini2.copy(); g.loc[(g.driver=='VER')&(g.lap_number==58),'lap_time_s'] = 1e9
up('huge2.csv', g.to_csv(index=False).encode(), 'last lap 58 = 1e9 s')
call('POST', '/api/forecast/latest', 'forecast huge2.csv VER', json={'file':'huge2.csv','driver':'VER'})
# duplicate laps
g = pd.concat([mini2, mini2[mini2.driver=='VER']])
up('dupes.csv', g.to_csv(index=False).encode(), 'duplicated VER rows')
call('POST', '/api/forecast/latest', 'forecast dupes.csv VER', json={'file':'dupes.csv','driver':'VER'})
call('POST', '/api/forecast/backtest', 'backtest dupes.csv VER', json={'file':'dupes.csv','driver':'VER','laps':5})
# one lap driver / two laps
g = mini2[(mini2.driver!='VER') | (mini2.lap_number<=1)]
up('onelap.csv', g.to_csv(index=False).encode(), 'VER has 1 lap')
call('POST', '/api/forecast/latest', 'forecast onelap.csv VER', json={'file':'onelap.csv','driver':'VER'})
g = mini2[(mini2.driver!='VER') | (mini2.lap_number<=2)]
up('twolaps.csv', g.to_csv(index=False).encode(), 'VER has 2 laps')
call('POST', '/api/forecast/latest', 'forecast twolaps.csv VER', json={'file':'twolaps.csv','driver':'VER'})
call('POST', '/api/forecast/backtest', 'backtest twolaps.csv VER', json={'file':'twolaps.csv','driver':'VER','laps':5})
# NaN driver, string season, unicode driver
g = mini2.copy(); g.loc[g.driver=='VER','driver'] = np.nan
up('nandriver.csv', g.to_csv(index=False).encode(), 'VER driver -> NaN')
call('POST', '/api/forecast/latest', 'forecast nandriver.csv default', json={'file':'nandriver.csv'})
g = mini2.copy(); g['season'] = 'twenty25'
up('strseason.csv', g.to_csv(index=False).encode(), 'season string')
call('POST', '/api/forecast/latest', 'forecast strseason.csv', json={'file':'strseason.csv','driver':'VER'})
g = mini2.copy(); g.loc[g.driver=='VER','driver'] = 'ВЕР'
up('unicode.csv', g.to_csv(index=False).encode(), 'cyrillic driver')
call('POST', '/api/forecast/latest', 'forecast unicode.csv', json={'file':'unicode.csv','driver':'ВЕР'})
# lap numbers reversed / gaps
g = mini2.copy(); g.loc[g.driver=='VER','lap_number'] = g.loc[g.driver=='VER','lap_number'] * 2
up('gaps.csv', g.to_csv(index=False).encode(), 'VER lap numbers doubled (gaps)')
call('POST', '/api/forecast/latest', 'forecast gaps.csv VER', json={'file':'gaps.csv','driver':'VER'})
call('POST', '/api/forecast/backtest', 'backtest gaps.csv VER', json={'file':'gaps.csv','driver':'VER','laps':5})
# session Q label and unknown event/team/circuit
g = mini2.copy(); g['session']='Q'
up('quali.csv', g.to_csv(index=False).encode(), 'session=Q relabel')
call('POST', '/api/forecast/latest', 'forecast quali.csv VER', json={'file':'quali.csv','driver':'VER'})
# multiple seasons latest pick: 2025 rows + a 1999 row later in file
g = pd.concat([mini2, pd.DataFrame([{'season':1999,'round_number':1,'event':'Old GP','session':'R','driver':'ZZZ','lap_number':1,'lap_time_s':100.0},{'season':1999,'round_number':1,'event':'Old GP','session':'R','driver':'ZZZ','lap_number':2,'lap_time_s':101.0}])])
up('mixed.csv', g.to_csv(index=False).encode(), 'mixed seasons')
call('POST', '/api/forecast/latest', 'forecast mixed.csv default', json={'file':'mixed.csv'})
print('== concurrency ==')
import concurrent.futures as cf
def one(i):
    t=time.time(); r=requests.post(B+'/api/forecast/backtest', json={'file':'f1_2025_season.parquet','driver':['VER','NOR','LEC','HAM','PIA'][i],'laps':20}, timeout=300); return r.status_code, round(time.time()-t,2)
with cf.ThreadPoolExecutor(5) as ex: print('5 parallel backtests:', list(ex.map(one, range(5))))
print('== summary endpoints ==')
call('GET', '/api/imports/f1_2025_season.parquet/summary', 'summary shipped')
call('GET', '/api/imports/minimal.csv/summary', 'summary minimal')
call('GET', '/api/datasets'); call('GET', '/api/models'); call('GET', '/api/experiments'); call('GET', '/api/reports/drivers'); call('GET', '/api/setup')
