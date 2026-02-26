export default function HomePage({ setPage }) {
  return (
    <div className="min-h-screen bg-[#070d0a] flex flex-col items-center justify-center text-center px-8 pt-20 pb-16 relative overflow-hidden">
      {/* Background grid */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(0,166,81,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(0,166,81,0.08) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
          maskImage:
            "radial-gradient(ellipse 80% 60% at 50% 40%, black, transparent)",
        }}
      />
      {/* Glow */}
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_80%_60%_at_50%_-10%,rgba(0,166,81,0.12)_0%,transparent_70%)]" />

      {/* Badge */}
      <div className="relative inline-flex items-center gap-2 bg-[#00A651]/10 border border-[#00A651]/25 rounded-full px-4 py-1.5 text-xs text-[#33C47A] font-medium tracking-widest uppercase mb-8">
        <span className="w-1.5 h-1.5 rounded-full bg-[#00A651] animate-pulse" />
        Live BDH Stream Active
      </div>

      {/* Title */}
      <h1
        className="text-5xl md:text-7xl font-bold leading-tight tracking-tight mb-6"
        style={{
          background: "linear-gradient(135deg, #fff 30%, #33C47A)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          fontFamily: "'Sora', sans-serif",
        }}
      >
        SSE Renewables<br />Climate Risk Platform
      </h1>

      <p className="text-[#8aad94] text-lg max-w-xl leading-relaxed mb-12">
        Real-time BDH physics model predictions, LLM-generated monthly risk reports, and an AI
        analyst — all grounded in SSE's corporate disclosures.
      </p>

      {/* CTAs */}
      <div className="flex gap-4 flex-wrap justify-center mb-16">
        <button
          onClick={() => setPage("live")}
          className="bg-[#00A651] hover:bg-[#007A3D] text-white px-8 py-3.5 rounded-lg text-sm font-semibold transition-all hover:-translate-y-0.5 font-[Sora] cursor-pointer"
        >
          → View Live Stream
        </button>
        <button
          onClick={() => setPage("analysis")}
          className="bg-transparent hover:bg-white/5 text-white border border-[#00A651]/30 hover:border-[#00A651] px-8 py-3.5 rounded-lg text-sm font-medium transition-all font-[Sora] cursor-pointer"
        >
          Monthly Reports
        </button>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-3 w-full max-w-xl border border-[#00A651]/18 rounded-2xl overflow-hidden divide-x divide-[#00A651]/18">
        {[
          { val: "210", unit: "MW", lbl: "Fleet Capacity" },
          { val: "84", unit: "", lbl: "BDH Features" },
          { val: "£98", unit: "/MWh", lbl: "CfD Strike Price" },
        ].map((s) => (
          <div key={s.lbl} className="bg-[#0f1c12] py-6 text-center">
            <div
              className="text-3xl font-bold text-[#00A651]"
              style={{ fontFamily: "'Space Mono', monospace" }}
            >
              {s.val}
              <span className="text-lg">{s.unit}</span>
            </div>
            <div className="text-[11px] text-[#8aad94] mt-1 uppercase tracking-widest">
              {s.lbl}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}