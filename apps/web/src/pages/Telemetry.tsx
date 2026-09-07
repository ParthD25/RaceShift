import { Activity } from 'lucide-react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Panel } from '../components/Panel';
import { CircuitMap } from '../components/CircuitMap';
import { SourceBadge } from '../components/SourceBadge';
import { telemetry } from '../data/mock';

export default function Telemetry() {
  return (
    <div className="page-stack">
      <div className="page-heading"><div><h1>Telemetry</h1><p>Speed, throttle and braking by distance. Telemetry summaries feed the model; the trace below is a fixture until a telemetry endpoint exists.</p></div><SourceBadge kind="fixture" /></div>
      <div className="two-col telemetry-layout">
        <Panel title="Track position" action={<SourceBadge kind="fixture" />}><CircuitMap /></Panel>
        <Panel title="Telemetry trace" icon={<Activity size={17} />} action={<SourceBadge kind="fixture" />}>
          <div className="big-chart">
            <ResponsiveContainer width="100%" height={390}>
              <LineChart data={telemetry}>
                <CartesianGrid stroke="#1f262d" vertical={false} />
                <XAxis dataKey="distance" tick={{ fill: '#81909c' }} />
                <YAxis tick={{ fill: '#81909c' }} />
                <Tooltip contentStyle={{ background: '#10161c', border: '1px solid #28313a' }} />
                <Line dataKey="speed" stroke="#edf3f8" dot={false} />
                <Line dataKey="throttle" stroke="#42d27a" dot={false} />
                <Line dataKey="brake" stroke="#ef4b50" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>
    </div>
  );
}
