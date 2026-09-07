import { useCallback, useEffect, useState } from 'react';
import { errorMessage } from './api';

export type ApiState<T> = { data: T | null; error: string | null; loading: boolean; reload: () => void };

/** Load one API resource on mount and expose a reload handle. */
export function useApi<T>(loader: () => Promise<T>, deps: unknown[] = []): ApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const reload = useCallback(() => setTick(t => t + 1), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    loader()
      .then(result => {
        if (cancelled) return;
        setData(result);
        setError(null);
      })
      .catch(err => {
        if (cancelled) return;
        setError(errorMessage(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, ...deps]);

  return { data, error, loading, reload };
}
