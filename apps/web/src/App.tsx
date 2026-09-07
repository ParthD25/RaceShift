import { Route, Routes } from 'react-router-dom';
import { AppShell } from './components/AppShell';
import Overview from './pages/Overview';
import Forecast from './pages/Forecast';
import Telemetry from './pages/Telemetry';
import Experiments from './pages/Experiments';
import Datasets from './pages/Datasets';
import Models from './pages/Models';
import SimplePage from './pages/SimplePage';

export default function App(){return <Routes><Route element={<AppShell/>}><Route index element={<Overview/>}/><Route path="forecast" element={<Forecast/>}/><Route path="telemetry" element={<Telemetry/>}/><Route path="experiments" element={<Experiments/>}/><Route path="datasets" element={<Datasets/>}/><Route path="models" element={<Models/>}/><Route path="compare" element={<SimplePage title="Compare Drivers" description="Compare pace, degradation and telemetry behavior under matched conditions."/>}/><Route path="strategy" element={<SimplePage title="Strategy Insights" description="Strategy analysis stays deterministic until the underlying pace and pit models are validated."/>}/><Route path="settings" element={<SimplePage title="Settings" description="Configure data sources, model artifacts and local-only application preferences."/>}/></Route></Routes>}
