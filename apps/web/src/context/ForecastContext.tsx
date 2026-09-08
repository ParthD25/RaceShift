import { createContext, useContext, useState, type ReactNode } from 'react';
import type { BacktestResult, ForecastResult } from '../lib/api';

type ForecastState = {
  result: ForecastResult | null;
  setResult: (result: ForecastResult | null) => void;
  backtest: BacktestResult | null;
  setBacktest: (backtest: BacktestResult | null) => void;
};

const ForecastContext = createContext<ForecastState>({
  result: null,
  setResult: () => undefined,
  backtest: null,
  setBacktest: () => undefined
});

export function ForecastProvider({ children }: { children: ReactNode }) {
  const [result, setResult] = useState<ForecastResult | null>(null);
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  return <ForecastContext.Provider value={{ result, setResult, backtest, setBacktest }}>{children}</ForecastContext.Provider>;
}

export function useForecast(): ForecastState {
  return useContext(ForecastContext);
}
