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

const KEY = 'raceshift.forecast.v1';

function restore(): { result: ForecastResult | null; backtest: BacktestResult | null; active: ActiveSession | null } {
  // Survive a page refresh (per tab). Storage can be unavailable or stale; fall back to empty.
  try {
    const raw = sessionStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return { result: parsed.result ?? null, backtest: parsed.backtest ?? null, active: parsed.active ?? null };
    }
  } catch { /* ignore */ }
  return { result: null, backtest: null, active: null };
}

export function ForecastProvider({ children }: { children: ReactNode }) {
  const initial = restore();
  const [result, setResult] = useState<ForecastResult | null>(initial.result);
  const [backtest, setBacktest] = useState<BacktestResult | null>(initial.backtest);
  const [active, setActive] = useState<ActiveSession | null>(initial.active);
  useEffect(() => {
    try { sessionStorage.setItem(KEY, JSON.stringify({ result, backtest, active })); } catch { /* ignore */ }
  }, [result, backtest, active]);
  return <ForecastContext.Provider value={{ result, setResult, backtest, setBacktest, active, setActive }}>{children}</ForecastContext.Provider>;
}

export function useForecast(): ForecastState {
  return useContext(ForecastContext);
}
