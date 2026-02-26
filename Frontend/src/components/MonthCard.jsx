import { useState } from "react";
import RiskBadge from "./RiskBadge";
import { SECTION_KEYS, SECTION_LABELS } from "../data/constants";

export default function MonthCard({ month: m }) {
  const [expanded, setExpanded] = useState(false);
  const [activeSection, setActiveSection] = useState(SECTION_KEYS[0]);

  return (
    <div
      className={`bg-[#0f1c12] border rounded-2xl overflow-hidden cursor-pointer transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_8px_32px_rgba(0,166,81,0.12)] ${
        expanded ? "border-[#00A651]" : "border-[#00A651]/18 hover:border-[#00A651]"
      }`}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Header */}
      <div className="px-5 py-4 flex items-center justify-between">
        <div>
          <span
            className="text-base font-bold text-white"
            style={{ fontFamily: "'Space Mono', monospace" }}
          >
            {m.month}
          </span>
          <div className="text-xs text-[#4d7a5a] mt-0.5">
            {m.key_numbers.wind_speed_avg_ms} m/s avg · 744h processed
          </div>
        </div>
        <div className="flex items-center gap-3">
          <RiskBadge rating={m.key_numbers.overall_risk_rating} />
          <span className="text-[#4d7a5a] text-sm">{expanded ? "↑" : "↓"}</span>
        </div>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-3 border-t border-[#00A651]/10 divide-x divide-[#00A651]/10">
        {[
          { v: `${m.key_numbers.capacity_factor_pct}%`, l: "Cap. Factor" },
          {
            v: `£${(m.key_numbers.est_revenue_cfd_gbp / 1e6).toFixed(2)}M`,
            l: "CfD Revenue",
          },
          {
            v: `${m.key_numbers.carbon_avoided_tco2e.toLocaleString()}`,
            l: "tCO2e Saved",
          },
        ].map((k) => (
          <div key={k.l} className="py-3 px-4">
            <div
              className="text-base font-bold text-white"
              style={{ fontFamily: "'Space Mono', monospace" }}
            >
              {k.v}
            </div>
            <div className="text-[10px] uppercase tracking-wider text-[#4d7a5a] mt-0.5">
              {k.l}
            </div>
          </div>
        ))}
      </div>

      {/* Headline */}
      <div className="px-5 py-3 border-t border-[#00A651]/10 text-xs text-[#8aad94] leading-relaxed">
        {m.llm_conclusion.headline}
      </div>

      {/* Expanded section */}
      {expanded && (
        <div
          className="px-5 pb-5 border-t border-[#00A651]/10"
          onClick={(e) => e.stopPropagation()}
        >
          {/* More KPIs */}
          <div className="grid grid-cols-3 gap-3 my-4">
            {[
              { l: "Gross Profit", v: `£${(m.key_numbers.est_gross_profit_gbp / 1e3).toFixed(0)}k` },
              { l: "Lost Revenue", v: `£${(m.key_numbers.lost_revenue_low_wind_gbp / 1e3).toFixed(0)}k` },
              { l: "Energy Generated", v: `${m.key_numbers.est_energy_mwh.toLocaleString()} MWh` },
              { l: "BDH Memory Norm", v: m.key_numbers.bdh_memory_norm_avg },
              { l: "High Wind Hours", v: `${m.key_numbers.high_wind_hours_gt12ms} hrs` },
              { l: "Low Wind Hours", v: `${m.key_numbers.low_wind_hours_lt4ms} hrs` },
            ].map((k) => (
              <div key={k.l} className="bg-[#070d0a] rounded-lg px-3 py-2.5">
                <div className="text-[10px] uppercase tracking-wider text-[#4d7a5a] mb-0.5">
                  {k.l}
                </div>
                <div
                  className="text-sm font-bold text-white"
                  style={{ fontFamily: "'Space Mono', monospace" }}
                >
                  {k.v}
                </div>
              </div>
            ))}
          </div>

          {/* Section tabs */}
          <div className="flex gap-1.5 flex-wrap mb-4">
            {SECTION_KEYS.map((k, i) => (
              <button
                key={k}
                onClick={() => setActiveSection(k)}
                className={`px-3 py-1 rounded-md text-[11px] font-medium cursor-pointer border transition-all font-[Sora] ${
                  activeSection === k
                    ? "bg-[#00A651]/15 text-[#00A651] border-[#00A651]/40"
                    : "bg-transparent text-[#8aad94] border-[#00A651]/18 hover:text-[#00A651] hover:border-[#00A651]"
                }`}
              >
                {SECTION_LABELS[i]}
              </button>
            ))}
          </div>

          {/* Section content */}
          <div className="text-xs text-[#8aad94] leading-7 whitespace-pre-line bg-[#070d0a] rounded-lg p-4">
            {m.llm_conclusion.sections[activeSection]}
          </div>
        </div>
      )}
    </div>
  );
}