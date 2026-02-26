import { MONTHLY_DATA } from "../data/constants";
import MonthCard from "../components/MonthCard";

export default function AnalysisPage() {
  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(MONTHLY_DATA, null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "sse_monthly_analysis.json";
    a.click();
  };

  const totalRevenue = MONTHLY_DATA.reduce(
    (s, m) => s + m.key_numbers.est_revenue_cfd_gbp,
    0
  );
  const totalCarbon = MONTHLY_DATA.reduce(
    (s, m) => s + m.key_numbers.carbon_avoided_tco2e,
    0
  );
  const highRiskCount = MONTHLY_DATA.filter(
    (m) => m.key_numbers.overall_risk_rating === "HIGH"
  ).length;

  return (
    <div className="pt-24 pb-16 px-6 max-w-7xl mx-auto">
      <h1
        className="text-3xl font-bold tracking-tight mb-1"
        style={{ fontFamily: "'Sora', sans-serif" }}
      >
        Monthly LLM Analysis
      </h1>
      <p className="text-[#8aad94] text-sm mb-8">
        BDH-grounded monthly reports — 6-section structured analysis per month
      </p>

      {/* Download button */}
      <button
        onClick={handleDownload}
        className="inline-flex items-center gap-2 bg-[#00A651]/10 border border-[#00A651] text-[#00A651] hover:bg-[#00A651] hover:text-white px-5 py-2.5 rounded-lg text-sm font-semibold cursor-pointer transition-all mb-8 font-[Sora]"
      >
        ↓ Download Full 24-Month Report
      </button>

      {/* Summary bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {[
          { label: "Total Months", value: "6", sub: "Jan–Jun 2023 (sample)", color: null },
          { label: "HIGH Risk Months", value: highRiskCount, sub: "Require action", color: "#ef4444" },
          { label: "Total Revenue", value: `£${(totalRevenue / 1e6).toFixed(1)}M`, sub: "CfD across 6 months", color: null },
          { label: "Carbon Avoided", value: `${(totalCarbon / 1e3).toFixed(1)}k`, sub: "tCO2e total", color: null },
        ].map((s) => (
          <div
            key={s.label}
            className="bg-[#0f1c12] border border-[#00A651]/18 rounded-xl px-5 py-4"
          >
            <div className="text-[11px] uppercase tracking-widest text-[#4d7a5a] mb-1">
              {s.label}
            </div>
            <div
              className="text-2xl font-bold"
              style={{
                fontFamily: "'Space Mono', monospace",
                color: s.color || "white",
              }}
            >
              {s.value}
            </div>
            <div className="text-xs text-[#4d7a5a] mt-0.5">{s.sub}</div>
          </div>
        ))}
      </div>

      {/* Month cards grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
        {MONTHLY_DATA.map((m) => (
          <MonthCard key={m.month} month={m} />
        ))}
      </div>
    </div>
  );
}