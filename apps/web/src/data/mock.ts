// Illustrative fixture content only. Everything in this file is generated or hand-written
// and is rendered with a "Fixture visual" badge. No value here is a model output or a
// measured Formula 1 result. Real numbers come from the local API (see lib/api.ts).

export const telemetry = Array.from({ length: 120 }, (_, i) => {
  const x = i / 119;
  const speed = 145 + 118 * Math.abs(Math.sin(x * Math.PI * 6.2)) + 35 * Math.sin(x * Math.PI * 17);
  const brakePulse = Math.max(0, Math.sin(x * Math.PI * 14 - 0.6));
  const throttle = Math.max(0, Math.min(100, 88 - brakePulse * 108 + 12 * Math.sin(x * Math.PI * 9)));
  const brake = Math.max(0, Math.min(100, brakePulse * 100 - 20));
  const rpm = 7.2 + Math.max(0, speed - 120) / 38 + 0.65 * Math.sin(x * Math.PI * 13);
  const delta = -0.18 * Math.sin(x * Math.PI * 4.4) + 0.08 * Math.sin(x * Math.PI * 11);
  return {
    distance: Number((x * 5.793).toFixed(3)),
    speed: Math.round(speed),
    throttle: Math.round(throttle),
    brake: Math.round(brake),
    rpm: Number(rpm.toFixed(2)),
    delta: Number(delta.toFixed(3))
  };
});

export const recentLaps = [
  { lap: 14, actual: '1:20.921', predicted: '1:21.104', delta: 0.183 },
  { lap: 13, actual: '1:20.845', predicted: '1:20.912', delta: 0.067 },
  { lap: 12, actual: '1:20.732', predicted: '1:20.689', delta: -0.043 },
  { lap: 11, actual: '1:21.021', predicted: '1:20.998', delta: -0.023 },
  { lap: 10, actual: '1:21.356', predicted: '1:21.205', delta: -0.151 },
  { lap: 9, actual: '1:21.112', predicted: '1:21.287', delta: 0.175 },
  { lap: 8, actual: '1:20.998', predicted: '1:21.046', delta: 0.048 }
];

export const factors = [
  { label: 'Tyre age', value: 24 },
  { label: 'Sector 2 traffic', value: 18 },
  { label: 'Track temperature', value: 15 },
  { label: 'Previous lap pace trend', value: 12 },
  { label: 'Corner 4 exit speed', value: 10 },
  { label: 'Fuel load estimate', value: 9 },
  { label: 'Driver consistency', value: 8 },
  { label: 'Headwind', value: 7 }
];
