import { createContext, useContext, useState, type ReactNode } from 'react';
import type { ForecastResult } from '../lib/api';

type ForecastState = {
  result: ForecastResult | null;
  setResult: (result: ForecastResult | null) => void;
};

const ForecastContext = createContext<ForecastState>({ result: null, setResult: () => undefined });

export function ForecastProvider({ children }: { children: ReactNode }) {
  const [result, setResult] = useState<ForecastResult | null>(null);
  return <ForecastContext.Provider value={{ result, setResult }}>{children}</ForecastContext.Provider>;
}

export function useForecast(): ForecastState {
  return useContext(ForecastContext);
}
