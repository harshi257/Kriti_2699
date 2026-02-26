import { useState } from "react";
import Navbar from "./components/Navbar";
import HomePage from "./pages/HomePage";
import LivePage from "./pages/LivePage";
import AnalysisPage from "./pages/AnalysisPage";
import ChatPage from "./pages/ChatPage";


export default function App() {
  const [page, setPage] = useState("home");
  return (
    <div className="min-h-screen bg-[#070d0a] text-[#e8f5ec]">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
        body, * { font-family: 'Sora', sans-serif; box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #070d0a; }
        ::-webkit-scrollbar-thumb { background: rgba(0,166,81,0.3); border-radius: 4px; }
      `}</style>
      <Navbar page={page} setPage={setPage} />
      {page === "home" && <HomePage setPage={setPage} />}
      {page === "live" && <LivePage />}
      {page === "analysis" && <AnalysisPage />}
      {page === "chat" && <ChatPage />}
    </div>
  );
}