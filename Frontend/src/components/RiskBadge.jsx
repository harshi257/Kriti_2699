export default function RiskBadge({ rating }) {
  const styles = {
    HIGH: "bg-red-500/15 text-red-400 border border-red-500/30",
    MEDIUM: "bg-amber-500/15 text-amber-400 border border-amber-500/30",
    LOW: "bg-[#00A651]/15 text-[#00A651] border border-[#00A651]/30",
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold tracking-wider ${
        styles[rating] || styles.MEDIUM
      }`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current" />
      {rating}
    </span>
  );
}