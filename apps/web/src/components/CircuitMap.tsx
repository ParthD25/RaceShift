const points = [
  { x: 72, y: 78, n: 1 }, { x: 113, y: 56, n: 2 }, { x: 168, y: 61, n: 3 },
  { x: 228, y: 95, n: 4 }, { x: 274, y: 127, n: 5 }, { x: 336, y: 132, n: 6 },
  { x: 398, y: 130, n: 7 }, { x: 452, y: 147, n: 8 }, { x: 468, y: 193, n: 9 },
  { x: 420, y: 219, n: 10 }, { x: 322, y: 205, n: 11 }
];

export function CircuitMap() {
  return (
    <div className="circuit-wrap">
      <svg viewBox="0 0 520 270" className="circuit" role="img" aria-label="Stylized Monza circuit map with pace delta segments">
        <path d="M69 79 C100 52 145 46 188 62 C226 76 251 108 282 125 C316 143 365 125 410 136 C453 146 486 164 468 195 C454 220 408 225 358 216 C302 207 274 185 228 176 C186 167 150 179 112 174 C73 168 49 144 50 112 C50 96 57 87 69 79Z" fill="none" stroke="#6d7782" strokeWidth="10" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M69 79 C100 52 145 46 188 62" fill="none" stroke="#49d67b" strokeWidth="10" strokeLinecap="round" />
        <path d="M188 62 C226 76 251 108 282 125" fill="none" stroke="#e94c4c" strokeWidth="10" strokeLinecap="round" />
        <path d="M282 125 C316 143 365 125 410 136" fill="none" stroke="#48d47b" strokeWidth="10" strokeLinecap="round" />
        <path d="M410 136 C453 146 486 164 468 195" fill="none" stroke="#e94c4c" strokeWidth="10" strokeLinecap="round" />
        <path d="M468 195 C454 220 408 225 358 216" fill="none" stroke="#e94c4c" strokeWidth="10" strokeLinecap="round" />
        <path d="M358 216 C302 207 274 185 228 176" fill="none" stroke="#48d47b" strokeWidth="10" strokeLinecap="round" />
        {points.map(p => <g key={p.n}><circle cx={p.x} cy={p.y} r="11" fill="#171d23" stroke="#7e8995" strokeWidth="1.5" /><text x={p.x} y={p.y + 4} textAnchor="middle" fontSize="9" fill="#e9eef3">{p.n}</text></g>)}
      </svg>
      <div className="legend"><span><i className="legend-dot gain" />Gain</span><span><i className="legend-dot neutral" />Neutral</span><span><i className="legend-dot loss" />Loss</span></div>
    </div>
  );
}
