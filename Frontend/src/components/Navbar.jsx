export default function Navbar({ page, setPage }) {
  const tabs = [
    { id: "live", label: "Live Stream" },
    { id: "analysis", label: "Monthly Analysis" },
    { id: "chat", label: "AI Analyst" },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-[#070d0a]/92 backdrop-blur-xl border-b border-[#00A651]/18 flex items-center justify-between px-8 h-16">
      {/* Logo */}
      <button
        onClick={() => setPage("home")}
        className="flex items-center gap-2.5 cursor-pointer"
      >
        <div className="w-8 h-8 bg-[#00A651] rounded-lg flex items-center justify-center text-lg">
          🌿
        </div>
        <div>
          <div className="font-bold text-sm tracking-wide text-white">SSE Renewables</div>
          <div className="text-[10px] text-[#8aad94] uppercase tracking-widest">
            Climate Risk Platform
          </div>
        </div>
      </button>

      {/* Tabs */}
      <div className="flex gap-1">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setPage(t.id)}
            className={`px-4 py-1.5 rounded-md text-[13px] font-medium cursor-pointer transition-all border font-[Sora] ${
              page === t.id
                ? "text-[#00A651] bg-[#00A651]/15 border-[#00A651]/30"
                : "text-[#8aad94] bg-transparent border-transparent hover:text-white hover:bg-white/5"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Status */}
      <div className="flex items-center gap-2 text-xs text-[#8aad94]">
        <span className="w-2 h-2 rounded-full bg-[#00A651] animate-pulse" />
        BDH Model Online
      </div>
    </nav>
  );
}