export default function Gauge({ value, max = 1, label }) {
  const pct = Math.min(value / max, 1);
  const r = 40, cx = 50, cy = 50;
  const arc = Math.PI * 1.5;
  const startAngle = Math.PI * 0.75;
  const endAngle = startAngle + arc * pct;
  const x1 = cx + r * Math.cos(startAngle), y1 = cy + r * Math.sin(startAngle);
  const x2 = cx + r * Math.cos(endAngle), y2 = cy + r * Math.sin(endAngle);
  const largeArc = arc * pct > Math.PI ? 1 : 0;
  const x1b = cx + r * Math.cos(Math.PI * 0.75 + arc);
  const y1b = cy + r * Math.sin(Math.PI * 0.75 + arc);

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 100 80" className="w-28 h-20">
        <path
          d={`M ${x1} ${y1} A ${r} ${r} 0 1 1 ${x1b} ${y1b}`}
          fill="none"
          stroke="rgba(0,166,81,0.15)"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {pct > 0 && (
          <path
            d={`M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`}
            fill="none"
            stroke="#00A651"
            strokeWidth="8"
            strokeLinecap="round"
          />
        )}
        <text
          x="50"
          y="54"
          textAnchor="middle"
          fill="#00A651"
          fontSize="13"
          fontWeight="700"
          fontFamily="'Space Mono', monospace"
        >
          {value.toFixed(2)}
        </text>
      </svg>
      <span className="text-xs uppercase tracking-widest text-[#4d7a5a] mt-1">{label}</span>
    </div>
  );
}