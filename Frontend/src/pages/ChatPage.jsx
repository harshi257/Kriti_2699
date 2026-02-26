import { useState, useRef, useEffect } from "react";
import { CHAT_HISTORY, MOCK_RESPONSES } from "../data/constants";

const TASKS = [
  { id: "qa", label: "Q&A", desc: "General questions" },
  { id: "risk_analysis", label: "Risk Analysis", desc: "Monthly operational report" },
  { id: "recommendation", label: "Recommendations", desc: "Actionable advice" },
  { id: "scenario", label: "Scenario Analysis", desc: "1.5°C / 2°C / 3°C" },
  { id: "esg", label: "ESG / TCFD", desc: "Governance & metrics" },
];

const QUICK = [
  "What are the main climate risks?",
  "What was the revenue in January?",
  "Which months were HIGH risk and why?",
  "How does SSE's CfD structure protect revenue?",
];

export default function ChatPage() {
  const [messages, setMessages] = useState(CHAT_HISTORY);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [task, setTask] = useState("qa");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = async (q) => {
    const question = q || input.trim();
    if (!question) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: question }]);
    setLoading(true);
    await new Promise((r) => setTimeout(r, 1400));
    const key = question.toLowerCase().includes("revenue")
      ? "revenue"
      : question.toLowerCase().includes("risk")
      ? "risk"
      : "default";
    const resp = MOCK_RESPONSES[key];
    setMessages((m) => [...m, { role: "assistant", text: resp.answer, sources: resp.sources }]);
    setLoading(false);
  };

  return (
    <div className="pt-24 pb-6 px-6 max-w-7xl mx-auto">
      <h1
        className="text-3xl font-bold tracking-tight mb-1"
        style={{ fontFamily: "'Sora', sans-serif" }}
      >
        AI Analyst Chat
      </h1>
      <p className="text-[#8aad94] text-sm mb-6">
        Query the LLM analyst — grounded in BDH model output and SSE corporate documents
      </p>

      <div className="grid grid-cols-1 md:grid-cols-[280px_1fr] gap-5 h-[calc(100vh-180px)]">
        {/* Sidebar */}
        <div className="flex flex-col gap-3">
          <div className="text-[11px] uppercase tracking-widest text-[#4d7a5a] mb-1 px-1">
            Analysis Mode
          </div>
          {TASKS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTask(t.id)}
              className={`text-left px-3.5 py-2.5 rounded-lg text-sm font-medium cursor-pointer border transition-all font-[Sora] ${
                task === t.id
                  ? "bg-[#00A651]/15 text-[#00A651] border-[#00A651]/40"
                  : "bg-[#0f1c12] text-[#8aad94] border-[#00A651]/18 hover:text-[#00A651] hover:border-[#00A651]"
              }`}
            >
              <div className="font-semibold">{t.label}</div>
              <div className="text-[11px] opacity-70 mt-0.5">{t.desc}</div>
            </button>
          ))}

          <div className="text-[11px] uppercase tracking-widest text-[#4d7a5a] mt-3 mb-1 px-1">
            Quick Questions
          </div>
          {QUICK.map((q) => (
            <button
              key={q}
              onClick={() => send(q)}
              className="text-left px-3.5 py-2 rounded-lg text-xs text-[#8aad94] hover:text-[#00A651] bg-[#0f1c12] border border-[#00A651]/18 hover:border-[#00A651] cursor-pointer transition-all font-[Sora]"
            >
              {q}
            </button>
          ))}
        </div>

        {/* Chat main */}
        <div className="flex flex-col bg-[#0f1c12] border border-[#00A651]/18 rounded-2xl overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-4 scrollbar-thin">
            {messages.map((m, i) => (
              <div
                key={i}
                className={`max-w-[80%] ${
                  m.role === "user" ? "self-end" : "self-start"
                }`}
              >
                <div
                  className={`px-4 py-3 rounded-xl text-sm leading-7 ${
                    m.role === "user"
                      ? "bg-[#00A651] text-white rounded-br-sm"
                      : "bg-[#111f15] border border-[#00A651]/18 text-[#e8f5ec] rounded-tl-sm"
                  }`}
                  style={{ whiteSpace: "pre-line" }}
                >
                  {m.text}
                </div>
                {m.sources && (
                  <div className="flex gap-2 flex-wrap mt-2">
                    {m.sources.map((s, j) => (
                      <span
                        key={j}
                        className="text-[10px] bg-[#070d0a] border border-[#00A651]/18 px-2 py-0.5 rounded text-[#4d7a5a]"
                        style={{ fontFamily: "'Space Mono', monospace" }}
                      >
                        {s.file} · p.{s.page}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="self-start bg-[#111f15] border border-[#00A651]/18 rounded-xl rounded-tl-sm px-4 py-3">
                <div className="flex gap-1 items-center">
                  {[0, 200, 400].map((delay) => (
                    <span
                      key={delay}
                      className="w-1.5 h-1.5 rounded-full bg-[#00A651] animate-bounce"
                      style={{ animationDelay: `${delay}ms` }}
                    />
                  ))}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="flex gap-3 px-5 py-4 border-t border-[#00A651]/18 bg-[#162419]">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !loading && send()}
              placeholder="Ask about wind performance, risk, or financials..."
              className="flex-1 bg-[#111f15] border border-[#00A651]/18 focus:border-[#00A651] outline-none rounded-lg px-4 py-2.5 text-sm text-[#e8f5ec] placeholder-[#4d7a5a] transition-colors font-[Sora]"
            />
            <button
              onClick={() => send()}
              disabled={loading || !input.trim()}
              className="bg-[#00A651] hover:bg-[#007A3D] disabled:opacity-40 disabled:cursor-not-allowed text-white px-5 py-2.5 rounded-lg text-sm font-semibold cursor-pointer transition-all font-[Sora]"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}