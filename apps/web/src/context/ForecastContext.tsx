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
const VERSION = 3;

type Stored = { result: ForecastResult | null; backtest: BacktestResult | null; active: ActiveSession | null };

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === 'object' && x !== null;
}
function looksLikeForecast(x: unknown): x is ForecastResult {
  return isRecord(x) && typeof x.predicted_next_lap_s === 'number' && typeof x.driver === 'string'
    && isRecord(x.context) && isRecord(x.historical_context);
}
function looksLikeBacktest(x: unknown): x is BacktestResult {
  return isRecord(x) && Array.isArray(x.laps) && isRecord(x.summary) && typeof x.driver === 'string';
}
function looksLikeActive(x: unknown): x is ActiveSession {
  // Every required field, so a truncated payload cannot put blanks in the top bar or label
  // unknown provenance as real.
  return isRecord(x) && typeof x.season === 'number' && typeof x.event === 'string' && typeof x.session === 'string'
    && typeof x.driver === 'string' && typeof x.artifact === 'string' && typeof x.is_synthetic === 'boolean';
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
