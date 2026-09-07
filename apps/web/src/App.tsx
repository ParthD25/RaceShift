import { Route, Routes } from 'react-router-dom';
import { AppShell } from './components/AppShell';
import { ForecastProvider } from './context/ForecastContext';
import Overview from './pages/Overview';
import Forecast from './pages/Forecast';
import Telemetry from './pages/Telemetry';
import Experiments from './pages/Experiments';
import Datasets from './pages/Datasets';
import Models from './pages/Models';
import Settings from './pages/Settings';
import SimplePage from './pages/SimplePage';

export default function App() {
  return (
    <ForecastProvider>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Overview />} />
          <Route path="forecast" element={<Forecast />} />
          <Route path="telemetry" element={<Telemetry />} />
          <Route path="experiments" element={<Experiments />} />
          <Route path="datasets" element={<Datasets />} />
          <Route path="models" element={<Models />} />
          <Route path="compare" element={<SimplePage title="Compare Drivers" description="Compare pace, degradation and telemetry behaviour under matched conditions." />} />
          <Route path="strategy" element={<SimplePage title="Strategy Insights" description="Strategy analysis stays deterministic until the underlying pace and pit models are validated." />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<SimplePage title="Page not found" description="This route does not exist in RaceShift. Use the navigation on the left." />} />
        </Route>
      </Routes>
    </ForecastProvider>
  );
}
