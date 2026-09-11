const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  const consoleErrors = [], failed = [], pageErrors = [];
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') consoleErrors.push(`[${m.type()}] ${m.text().slice(0, 200)}`); });
  page.on('pageerror', e => pageErrors.push(String(e).slice(0, 300)));
  page.on('response', r => { if (r.status() >= 400) failed.push(`${r.status()} ${r.request().method()} ${r.url()}`); });
  const base = 'http://127.0.0.1:5173';
  const routes = ['/', '/forecast', '/compare', '/experiments', '/datasets', '/models', '/settings', '/telemetry', '/strategy', '/does-not-exist'];
  for (const r of routes) {
    const t = Date.now();
    try {
      await page.goto(base + r, { waitUntil: 'networkidle', timeout: 60000 });
      await page.waitForTimeout(1500);
      const h1 = await page.locator('h1').first().textContent().catch(() => null);
      const text = (await page.locator('body').innerText()).replace(/\s+/g, ' ');
      console.log(`ROUTE ${r} -> h1=${JSON.stringify(h1)} (${Date.now() - t}ms) errors_on_page=${/error|failed|unavailable/i.test(text) ? 'maybe' : 'no'} len=${text.length}`);
      await page.screenshot({ path: `shots${r === '/' ? '/home' : r.replace(/\//g, '_')}.png`, fullPage: true });
    } catch (e) { console.log(`ROUTE ${r} FAILED: ${String(e).slice(0, 200)}`); }
  }
  // Forecast flow on 2026 data
  await page.goto(base + '/forecast', { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(1500);
  const selects = page.locator('select');
  console.log('select count', await selects.count());
  const dsOptions = await selects.nth(0).locator('option').allTextContents();
  console.log('dataset options:', dsOptions.join(' | '));
  await selects.nth(0).selectOption('f1_2026_races.parquet');
  await page.waitForTimeout(2500);
  const drvOptions = await selects.nth(1).locator('option').allTextContents();
  console.log('driver options after selecting 2026 file:', drvOptions.slice(0, 8).join(' | '), '... total', drvOptions.length);
  const modelOptions = await selects.nth(2).locator('option').allTextContents();
  console.log('model options:', modelOptions.join(' | '));
  const summaryText = (await page.locator('body').innerText()).replace(/\s+/g, ' ');
  const m = summaryText.match(/Rows.{0,40}Seasons.{0,60}Latest session.{0,80}Provenance.{0,40}/); console.log('summary block:', m ? m[0] : 'NOT FOUND');
  const t0 = Date.now();
  await page.getByRole('button', { name: /Run forecast/ }).click();
  try { await page.getByText(/How did it do on/).waitFor({ timeout: 120000 }); } catch (e) { console.log('backtest panel did not appear:', String(e).slice(0, 100)); }
  await page.waitForTimeout(3000);
  console.log(`forecast+backtest UI round trip ${Date.now() - t0}ms`);
  const body = (await page.locator('body').innerText()).replace(/\s+/g, ' ');
  const idx = body.indexOf('How did it do'); console.log('RESULT TEXT:', body.slice(idx, idx + 1600));
  const nextIdx = body.indexOf('Next lap forecast'); console.log('FORECAST TEXT:', body.slice(nextIdx, nextIdx + 900));
  await page.screenshot({ path: 'shots/forecast_2026_result.png', fullPage: true });
  // Select a driver with a DNF (few laps) if present, e.g. first driver option, run again
  await selects.nth(1).selectOption({ index: 1 });
  await page.getByRole('button', { name: /Run forecast/ }).click();
  await page.waitForTimeout(4000);
  const body2 = (await page.locator('body').innerText()).replace(/\s+/g, ' ');
  const i2 = body2.indexOf('How did it do'); console.log('SECOND DRIVER RESULT:', body2.slice(i2, i2 + 500));
  const errBoxes = await page.locator('.error-box, .warn-box').allTextContents(); console.log('error/warn boxes:', JSON.stringify(errBoxes).slice(0, 800));
  // Compare page with 2026 file
  await page.goto(base + '/compare', { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(1500);
  const cmpSelects = page.locator('select'); console.log('compare selects', await cmpSelects.count());
  try {
    await cmpSelects.nth(0).selectOption('f1_2026_races.parquet'); await page.waitForTimeout(2500);
    const btn = page.getByRole('button').filter({ hasText: /Compare|Run/ }).first(); console.log('compare button:', await btn.textContent());
    await btn.click(); await page.waitForTimeout(15000);
    const cb = (await page.locator('body').innerText()).replace(/\s+/g, ' '); console.log('COMPARE TEXT:', cb.slice(0, 1200));
    await page.screenshot({ path: 'shots/compare_2026.png', fullPage: true });
  } catch (e) { console.log('compare flow error', String(e).slice(0, 200)); }
  console.log('CONSOLE ERRORS/WARNINGS:', JSON.stringify(consoleErrors.slice(0, 20), null, 1));
  console.log('PAGE ERRORS:', JSON.stringify(pageErrors));
  console.log('FAILED RESPONSES:', JSON.stringify(failed.slice(0, 20)));
  await browser.close();
})().catch(e => { console.error('FATAL', e); process.exit(1); });
