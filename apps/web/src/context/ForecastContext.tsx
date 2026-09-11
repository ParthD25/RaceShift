import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import type { BacktestResult, ForecastResult } from '../lib/api';

// What the top bar shows: the session and driver the user last worked with, set by the
// Forecast page (from its result) or by Compare Drivers (from its two backtests).
export type ActiveSession = {
  season: number; event: string; session: string; driver: string; artifact: string;
  is_synthetic: boolean; data_is_synthetic?: boolean;
};

type ForecastState = {
  result: ForecastResult | null;
  setResult: (result: ForecastResult | null) => void;
  backtest: BacktestResult | null;
  setBacktest: (backtest: BacktestResult | null) => void;
  active: ActiveSession | null;
  setActive: (active: ActiveSession | null) => void;
};

const ForecastContext = createContext<ForecastState>({
  result: null, setResult: () => undefined,
  backtest: null, setBacktest: () => undefined,
  active: null, setActive: () => undefined
});

// Bump when the shape of ForecastResult/BacktestResult/ActiveSession changes so a payload
// saved by an older build is discarded instead of dereferenced.
const KEY = 'raceshift.forecast';
const VERSION = 2;

type Stored = { result: ForecastResult | null; backtest: BacktestResult | null; active: ActiveSession | null };

function looksLikeForecast(x: unknown): x is ForecastResult {
  return typeof x === 'object' && x !== null && typeof (x as ForecastResult).predicted_next_lap_s === 'number'
    && typeof (x as ForecastResult).driver === 'string' && typeof (x as ForecastResult).context === 'object'
    && typeof (x as ForecastResult).historical_context === 'object';
}
function looksLikeBacktest(x: unknown): x is BacktestResult {
  return typeof x === 'object' && x !== null && Array.isArray((x as BacktestResult).laps)
    && typeof (x as BacktestResult).summary === 'object' && typeof (x as BacktestResult).driver === 'string';
}
function looksLikeActive(x: unknown): x is ActiveSession {
  return typeof x === 'object' && x !== null && typeof (x as ActiveSession).event === 'string' && typeof (x as ActiveSession).driver === 'string';
}

function restore(): Stored {
  // Survive a page refresh (per tab). Storage can be unavailable, stale or from an older
  // build; anything that does not pass the shape checks is dropped.
  try {
    const raw = sessionStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && parsed.v === VERSION) {
        return {
          result: looksLikeForecast(parsed.result) ? parsed.result : null,
          backtest: looksLikeBacktest(parsed.backtest) ? parsed.backtest : null,
          active: looksLikeActive(parsed.active) ? parsed.active : null
        };
      }
    }
  } catch { /* ignore */ }
  return { result: null, backtest: null, active: null };
}

export function ForecastProvider({ children }: { children: ReactNode }) {
  const [initial] = useState<Stored>(restore);
  const [result, setResult] = useState<ForecastResult | null>(initial.result);
  const [backtest, setBacktest] = useState<BacktestResult | null>(initial.backtest);
  const [active, setActive] = useState<ActiveSession | null>(initial.active);
  useEffect(() => {
    try { sessionStorage.setItem(KEY, JSON.stringify({ v: VERSION, result, backtest, active })); } catch { /* ignore */ }
  }, [result, backtest, active]);
  return <ForecastContext.Provider value={{ result, setResult, backtest, setBacktest, active, setActive }}>{children}</ForecastContext.Provider>;
}

export function useForecast(): ForecastState {
  return useContext(ForecastContext);
}
