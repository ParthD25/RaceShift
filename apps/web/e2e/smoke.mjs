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

// Forecast: default selection must be the shipped real season and produce a forecast + backtest.
await page.goto(`${web}/forecast`, { waitUntil: 'networkidle' });
await page.waitForSelector('select');
await page.waitForFunction(() => document.querySelectorAll('select')[0]?.value, null, { timeout: 15000 });
const dataset = await page.inputValue('select >> nth=0');
check(dataset === 'f1_2025_season.parquet', `default dataset is ${dataset}, expected f1_2025_season.parquet`);
await page.click('button.primary-btn');
await page.waitForSelector('.forecast-time', { timeout: 90000 });
const lap = await page.textContent('.forecast-time');
check(/^\d:\d\d\.\d{3}$/.test(lap ?? ''), `forecast time renders as ${lap}`);
await page.waitForSelector('.backtest-table .wide-row', { timeout: 90000 }).catch(() => failures.push('backtest table did not render'));
const rows = await page.locator('.backtest-table .wide-row').count();
check(rows >= 1, `backtest rows: ${rows}`);
await shot('forecast');

// Overview: only API-backed panels, and the forecast just run is shown. Navigate in-app (a
// full page load would reset the session state on purpose).
await page.click('a.nav-item[href="/"]');
await page.waitForSelector('.dashboard-grid', { timeout: 15000 });
await page.waitForTimeout(500);
check((await page.locator('.forecast-time').count()) === 1, 'overview shows the forecast that was run');
check((await page.locator('.badge-fixture').count()) === 0, 'overview has no fixture-visual badges');
await shot('overview');

// Compare Drivers: two backtests side by side.
await page.goto(`${web}/compare`, { waitUntil: 'networkidle' });
await page.waitForSelector('select');
await page.waitForFunction(() => document.querySelectorAll('select')[1]?.value, null, { timeout: 15000 });
await page.click('button.primary-btn');
await page.waitForSelector('.compare-grid .panel', { timeout: 120000 }).catch(() => failures.push('compare grid did not render'));
check((await page.locator('.compare-grid .panel').count()) === 2, 'compare shows two driver panels');
await shot('compare');

// Every other route renders without throwing.
for (const route of ['experiments', 'datasets', 'models', 'settings', 'telemetry', 'strategy', 'nope']) {
  await page.goto(`${web}/${route}`, { waitUntil: 'networkidle' });
  check((await page.locator('h1').count()) >= 1, `${route} renders a heading`);
}

await browser.close();
if (failures.length) {
  console.error('E2E FAILED');
  for (const f of failures) console.error(' -', f);
  process.exit(1);
}
console.log(`E2E OK: dataset=${dataset} forecast=${lap} backtestRows=${rows}`);
