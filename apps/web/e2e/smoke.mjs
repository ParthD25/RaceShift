// End-to-end smoke test: drives the real UI against the real local API with Playwright.
//
//   API_PORT=8000 WEB_PORT=5173 node apps/web/e2e/smoke.mjs
//
// Expects `npm run dev` (or the API and Vite dev server) to be up. Fails (exit 1) when a page
// throws, when the forecast or backtest does not render, or when a fixture-only panel appears
// on a page that must be API-backed. Screenshots land in the directory named by E2E_SHOTS.
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const web = `http://127.0.0.1:${process.env.WEB_PORT ?? 5173}`;
const shots = process.env.E2E_SHOTS ?? '';
if (shots) mkdirSync(shots, { recursive: true });
const failures = [];
const check = (ok, message) => { if (!ok) failures.push(message); };

const browser = await chromium.launch({ executablePath: process.env.PW_EXEC || undefined });
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
page.on('pageerror', e => failures.push(`page error: ${e.message}`));
page.on('console', m => { if (m.type() === 'error') failures.push(`console error: ${m.text().slice(0, 200)}`); });
const shot = async name => { if (shots) await page.screenshot({ path: `${shots}/${name}.png`, fullPage: true }); };
// Record a failed wait instead of throwing, so one missing element does not abort the run.
const waitFor = async (selector, timeout, message) => {
  try { await page.waitForSelector(selector, { timeout }); return true; } catch { failures.push(message); return false; }
};
// The Run buttons stay disabled until the dataset, session, driver and artifact selects have
// loaded from separate API calls; wait for the button itself rather than for one select.
const waitEnabled = async (timeout, message) => {
  try { await page.waitForFunction(() => { const b = document.querySelector('button.primary-btn'); return b && !b.disabled; }, null, { timeout }); return true; } catch { failures.push(message); return false; }
};

let dataset = '';
let lap = '';
let rows = 0;
try {
  // Forecast: default selection must be the shipped real season and produce a forecast + backtest.
  await page.goto(`${web}/forecast`, { waitUntil: 'networkidle' });
  await waitFor('select', 15000, 'forecast page shows no selects');
  await page.waitForFunction(() => document.querySelectorAll('select')[0]?.value, null, { timeout: 15000 }).catch(() => failures.push('dataset select never got a value'));
  dataset = await page.inputValue('select >> nth=0').catch(() => '');
  check(dataset === 'f1_2025_season.parquet', `default dataset is ${dataset}, expected f1_2025_season.parquet`);
  if (await waitEnabled(30000, 'forecast Run button never became enabled')) {
    await page.click('button.primary-btn');
    if (await waitFor('.forecast-time', 90000, 'forecast did not render')) {
      lap = (await page.textContent('.forecast-time')) ?? '';
      check(/^\d:\d\d\.\d{3}$/.test(lap), `forecast time renders as ${lap}`);
    }
    if (await waitFor('.backtest-table .wide-row', 90000, 'backtest table did not render')) {
      rows = await page.locator('.backtest-table .wide-row').count();
      check(rows >= 1, `backtest rows: ${rows}`);
    }
  }
  await shot('forecast');

  // Overview: only API-backed panels, and the forecast just run is shown. Navigate in-app (a
  // full page load would reset the session state on purpose).
  await page.click('a.nav-item[href="/"]');
  await waitFor('.dashboard-grid', 15000, 'overview grid did not render');
  await page.waitForTimeout(500);
  check((await page.locator('.forecast-time').count()) === 1, 'overview shows the forecast that was run');
  check((await page.locator('.badge-fixture').count()) === 0, 'overview has no fixture-visual badges');
  await shot('overview');

  // Compare Drivers: two backtests side by side.
  await page.goto(`${web}/compare`, { waitUntil: 'networkidle' });
  await waitFor('select', 15000, 'compare page shows no selects');
  if (await waitEnabled(30000, 'compare Run button never became enabled')) {
    await page.click('button.primary-btn');
    if (await waitFor('.compare-grid .panel', 120000, 'compare grid did not render')) {
      check((await page.locator('.compare-grid .panel').count()) === 2, 'compare shows two driver panels');
    }
  }
  await shot('compare');

  // Every other route renders without throwing; unknown routes render the not-found page.
  for (const route of ['experiments', 'datasets', 'models', 'settings', 'nope']) {
    await page.goto(`${web}/${route}`, { waitUntil: 'networkidle' });
    check((await page.locator('h1').count()) >= 1, `${route} renders a heading`);
  }
} catch (err) {
  failures.push(`unexpected error: ${err instanceof Error ? err.message : String(err)}`);
} finally {
  await browser.close();
}
if (failures.length) {
  console.error('E2E FAILED');
  for (const f of failures) console.error(' -', f);
  process.exit(1);
}
console.log(`E2E OK: dataset=${dataset} forecast=${lap} backtestRows=${rows}`);
