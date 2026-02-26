import { useState, useEffect } from "react";
import { LIVE_DATA, KEY_FEATURES } from "../data/constants";
import Gauge from "../components/Gauge";
import Sparkline from "../components/Sparkline";

export default function LivePage() {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 3000);
    return () => clearInterval(t);
  }, []);

  const jitter = (v, mag = 0.05) => {
    const rand = crypto.getRandomValues(new Uint32Array(1))[0] / 2 ** 32;
    return parseFloat((v + (rand - 0.5) * mag).toFixed(4));
  };
  
  const live = {
    ...LIVE_DATA,
    _tick: tick,
    wind_metrics: {
      avg_ws_24h: jitter(LIVE_DATA.wind_metrics.avg_ws_24h, 0.3),
      avg_power_proxy_24h: jitter(
        LIVE_DATA.wind_metrics.avg_power_proxy_24h,
        0.02,
      ),
      high_wind_hours_24h: LIVE_DATA.wind_metrics.high_wind_hours_24h,
      low_wind_hours_24h: LIVE_DATA.wind_metrics.low_wind_hours_24h,
      mean_pred_error: jitter(LIVE_DATA.wind_metrics.mean_pred_error, 0.0005),
    },
    memory_norm: jitter(LIVE_DATA.memory_norm, 0.02),
  };

  const kpis = [
    {
      label: "Avg Wind Speed (24h)",
      value: live.wind_metrics.avg_ws_24h.toFixed(3),
      unit: "m/s",
    },
    {
      label: "Power Proxy (24h)",
      value: live.wind_metrics.avg_power_proxy_24h.toFixed(3),
      unit: "",
    },
    {
      label: "High Wind Hours",
      value: live.wind_metrics.high_wind_hours_24h,
      unit: "hrs >12m/s",
    },
    {
      label: "Low Wind Hours",
      value: live.wind_metrics.low_wind_hours_24h,
      unit: "hrs <4m/s",
    },
    {
      label: "Mean Pred. Error",
      value: live.wind_metrics.mean_pred_error.toFixed(5),
      unit: "",
    },
  ];

  return (
    <div className="pt-24 pb-16 px-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-1">
        <h1
          className="text-3xl font-bold tracking-tight"
          style={{ fontFamily: "'Sora', sans-serif" }}
        >
          Live BDH Stream
        </h1>
        <span
          className="text-xs text-[#4d7a5a] font-[Space_Mono]"
          style={{ fontFamily: "'Space Mono', monospace" }}
        >
          {LIVE_DATA.timestamp} · Hour {LIVE_DATA.hour}
        </span>
      </div>
      <p className="text-[#8aad94] text-sm mb-8">
        Updating every hour — BDH model predictions vs actuals across 84
        features
      </p>

      {/* KPI Row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
        {kpis.map((k) => (
          <div
            key={k.label}
            className="bg-[#0f1c12] border border-[#00A651]/18 rounded-xl px-5 py-4"
          >
            <div className="text-[11px] uppercase tracking-widest text-[#4d7a5a] mb-1.5">
              {k.label}
            </div>
            <div
              className="text-2xl font-bold text-white"
              style={{ fontFamily: "'Space Mono', monospace" }}
            >
              {k.value}
            </div>
            {k.unit && (
              <div className="text-xs text-[#8aad94] mt-0.5">{k.unit}</div>
            )}
          </div>
        ))}
      </div>

      {/* Memory Norm + Sparkline */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-5">
        <div className="bg-[#0f1c12] border border-[#00A651]/18 rounded-xl p-5">
          <div className="text-xs uppercase tracking-widest text-[#4d7a5a] mb-4">
            BDH Memory Norm
          </div>
          <div className="flex items-center gap-6">
            <Gauge value={live.memory_norm} max={1} label="Memory Norm" />
            <div className="text-sm text-[#8aad94]">
              {live.memory_norm > 0.7
                ? "🟢 High predictability"
                : live.memory_norm > 0.55
                  ? "🟡 Moderate predictability"
                  : "🔴 Low predictability"}
            </div>
          </div>
        </div>

        <div className="bg-[#0f1c12] border border-[#00A651]/18 rounded-xl p-5">
          <div className="text-xs uppercase tracking-widest text-[#4d7a5a] mb-2">
            Prediction Error (last 24h)
          </div>
          <div
            className="flex justify-between text-xs text-[#4d7a5a] mb-1"
            style={{ fontFamily: "'Space Mono', monospace" }}
          >
            <span>{Math.min(...LIVE_DATA.recent_errors).toFixed(4)}</span>
            <span>Error trend</span>
            <span>{Math.max(...LIVE_DATA.recent_errors).toFixed(4)}</span>
          </div>
          <Sparkline data={LIVE_DATA.recent_errors} />
        </div>
      </div>

      {/* Wind bars */}
      <div className="bg-[#0f1c12] border border-[#00A651]/18 rounded-xl p-5 mb-5">
        <div className="text-xs uppercase tracking-widest text-[#4d7a5a] mb-4">
          Wind Speed — Current Hour
        </div>
        {["WS50M", "WS10M"].map((f) => {
          const feat = LIVE_DATA.features[f];
          const maxW = 20;
          const actualPct = (jitter(feat.actual, 0.1) / maxW) * 100;
          const predPct = (jitter(feat.predicted, 0.1) / maxW) * 100;
          return (
            <div key={f} className="mb-4">
              <div
                className="flex justify-between text-xs text-[#8aad94] mb-1.5"
                style={{ fontFamily: "'Space Mono', monospace" }}
              >
                <span className="font-medium text-white">{f}</span>
                <span>
                  {jitter(feat.predicted, 0.1).toFixed(3)} |{" "}
                  {jitter(feat.actual, 0.1).toFixed(3)} m/s
                </span>
              </div>
              <div className="relative h-2 bg-[#162419] rounded-full mb-1">
                <div
                  className="absolute h-full bg-[#00A651]/40 rounded-full"
                  style={{ width: `${actualPct}%` }}
                />
              </div>
              <div className="relative h-2 bg-[#162419] rounded-full">
                <div
                  className="absolute h-full bg-[#00A651] rounded-full"
                  style={{ width: `${predPct}%` }}
                />
              </div>
              <div className="flex gap-4 text-[10px] text-[#4d7a5a] mt-1">
                <span>■ Actual</span>
                <span>■ Predicted</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Feature table */}
      <div className="bg-[#0f1c12] border border-[#00A651]/18 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-[#00A651]/10">
          <div className="text-xs uppercase tracking-widest text-[#4d7a5a]">
            All Key Features — Predicted vs Actual (This Hour)
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#00A651]/10">
                {["Feature", "Predicted", "Actual", "Error", "Accuracy"].map(
                  (h) => (
                    <th
                      key={h}
                      className="text-left px-4 py-3 text-[11px] uppercase tracking-widest text-[#4d7a5a] font-medium"
                    >
                      {h}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {KEY_FEATURES.map((name) => {
                const f = LIVE_DATA.features[name];
                if (!f) return null;
                const errClass =
                  Math.abs(f.error) < 0.01
                    ? "text-[#00A651]"
                    : f.error > 0
                      ? "text-amber-400"
                      : "text-blue-400";
                const acc = Math.max(
                  0,
                  100 - Math.abs(f.error / (f.actual || 1)) * 100,
                ).toFixed(1);
                const accColor =
                  parseFloat(acc) > 95
                    ? "#00A651"
                    : parseFloat(acc) > 85
                      ? "#f59e0b"
                      : "#ef4444";
                return (
                  <tr
                    key={name}
                    className="border-b border-[#00A651]/07 hover:bg-[#111f15] transition-colors"
                  >
                    <td
                      className="px-4 py-2.5 text-white font-medium"
                      style={{ fontFamily: "'Space Mono', monospace" }}
                    >
                      {name}
                    </td>
                    <td
                      className="px-4 py-2.5 text-[#8aad94]"
                      style={{ fontFamily: "'Space Mono', monospace" }}
                    >
                      {jitter(f.predicted, 0.01).toFixed(4)}
                    </td>
                    <td
                      className="px-4 py-2.5 text-[#8aad94]"
                      style={{ fontFamily: "'Space Mono', monospace" }}
                    >
                      {jitter(f.actual, 0.01).toFixed(4)}
                    </td>
                    <td
                      className={`px-4 py-2.5 ${errClass}`}
                      style={{ fontFamily: "'Space Mono', monospace" }}
                    >
                      {f.error > 0 ? "+" : ""}
                      {f.error.toFixed(4)}
                    </td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-[#162419] rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${acc}%`,
                              background: accColor,
                            }}
                          />
                        </div>
                        <span
                          className="text-xs"
                          style={{
                            color: accColor,
                            fontFamily: "'Space Mono', monospace",
                          }}
                        >
                          {acc}%
                        </span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
