"""
pipeline.py — SSE Renewables BDH Climate Risk Pipeline
Extracted from notebook cells 2–6 for standalone Render deployment.
"""

import os
import re
import math
import json
import dataclasses
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from groq import Groq
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Global Config
# ─────────────────────────────────────────────────────────────────────────────
LLM_MODEL         = "llama-3.3-70b-versatile"
MAX_TOKENS        = 1024
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 150
RETRIEVAL_TOP_K   = 5
REPORTS_DIR       = os.environ.get("REPORTS_DIR", "/app/reports")
CHROMA_DIR        = os.environ.get("CHROMA_DIR",  "/app/chroma_db")
FEATURE_DIM       = 84
SEQ_LEN           = 32
DEVICE            = "cpu"   # CPU on Render free tier

from langchain_core.documents import Document

DEMO_DOCS = [
    Document(
        page_content=(
            "SSE targets net zero by 2050 with an 80% emissions reduction by 2030. "
            "The company has committed to investing £18bn in low-carbon infrastructure "
            "over the next 5 years, focused on wind, solar and electricity networks."
        ),
        metadata={"source": "demo", "source_file": "demo_sse_strategy.txt",
                  "report_year": "2023", "page": 1}
    ),
    Document(
        page_content=(
            "SSE's TCFD report highlights physical risks including increased storm frequency, "
            "wind resource variability, and flooding at coastal assets. Transition risks include "
            "carbon pricing, policy changes, and technology disruption. SSE has embedded climate "
            "risk into its enterprise risk management framework."
        ),
        metadata={"source": "demo", "source_file": "demo_tcfd_report.txt",
                  "report_year": "2023", "page": 1}
    ),
    Document(
        page_content=(
            "Dogger Bank Wind Farm (3.6GW) is a joint venture between SSE Renewables, Equinor, "
            "and Vårgrønn. Phase A and B are under construction with first power expected in 2023. "
            "Seagreen (1.075GW, offshore Scotland) reached full commercial operations in 2023. "
            "Viking Energy (443MW, Shetland) is SSE's largest onshore wind farm."
        ),
        metadata={"source": "demo", "source_file": "demo_asset_overview.txt",
                  "report_year": "2023", "page": 1}
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — BDH Model Definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 4
    n_embd:  int = 128
    dropout: float = 0.1
    n_head:  int = 4
    mlp_internal_dim_multiplier: int = 16
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (
        1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        nh = config.n_head
        D  = config.n_embd
        N  = config.mlp_internal_dim_multiplier * D // nh
        self.register_buffer(
            "freqs",
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = Attention.phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        QR     = self.rope(r_phases, Q)
        scores = (QR @ QR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        self.config = config
        nh = config.n_head
        D  = config.n_embd
        N  = D * config.mlp_internal_dim_multiplier // nh
        self.input_proj = nn.Linear(input_dim, D)
        self.decoder    = nn.Parameter(torch.zeros(nh * N, D).normal_(std=0.02))
        self.encoder    = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
        self.encoder_v  = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
        self.attn = Attention(config)
        self.ln   = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)
        self.head = nn.Linear(D, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, targets=None):
        B, T, _ = x.size()
        D  = self.config.n_embd
        nh = self.config.n_head
        N  = D * self.config.mlp_internal_dim_multiplier // nh
        x  = self.input_proj(x)
        x  = self.ln(x).unsqueeze(1)
        for _ in range(self.config.n_layer):
            x_res     = x
            x_latent  = x @ self.encoder
            x_sparse  = F.relu(x_latent)
            yKV       = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV       = self.ln(yKV)
            y_latent  = yKV @ self.encoder_v
            y_sparse  = F.relu(y_latent)
            xy_sparse = self.drop(x_sparse * y_sparse)
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )
            x = self.ln(x_res + self.ln(yMLP))
        out = x.squeeze(1)
        logits = self.head(out)
        loss = None
        if targets is not None:
            loss = F.mse_loss(logits, targets)
        return logits, loss

    def get_memory_norm(self, x):
        """Extract normalised latent memory from the last layer."""
        B, T, _ = x.size()
        D  = self.config.n_embd
        nh = self.config.n_head
        N  = D * self.config.mlp_internal_dim_multiplier // nh
        xh = self.input_proj(x)
        xh = self.ln(xh).unsqueeze(1)
        for _ in range(self.config.n_layer):
            x_res     = xh
            x_latent  = xh @ self.encoder
            x_sparse  = F.relu(x_latent)
            yKV       = self.attn(Q=x_sparse, K=x_sparse, V=xh)
            yKV       = self.ln(yKV)
            y_latent  = yKV @ self.encoder_v
            y_sparse  = F.relu(y_latent)
            xy_sparse = self.drop(x_sparse * y_sparse)
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )
            xh = self.ln(x_res + self.ln(yMLP))
        memory = xh.squeeze(1)[:, -1, :]
        return float(memory.norm(dim=-1).mean().item())


def run_inference(model, window_np):
    """Run BDH on a single SEQ_LEN window; returns (predictions, memory_norm)."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(window_np).float().unsqueeze(0).to(DEVICE)
        logits, _ = model(x)
        mem_norm  = model.get_memory_norm(x)
    return logits.squeeze(0).cpu().numpy(), mem_norm


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — RAG Vector Store
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(reports_dir):
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    docs      = []
    pdf_files = list(Path(reports_dir).glob("**/*.pdf"))
    csv_files = list(Path(reports_dir).glob("**/*.csv"))
    all_files = pdf_files + csv_files

    if not all_files:
        logger.warning("No PDF/CSV files found — using DEMO documents.")
        return DEMO_DOCS

    for file_path in all_files:
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path), encoding="utf-8")
            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata["source_file"] = file_path.name
                doc.metadata["file_type"]   = file_path.suffix.lower()
                m = re.search(r"(20\d{2})", file_path.name)
                doc.metadata["report_year"] = m.group(1) if m else "unknown"
            docs.extend(file_docs)
        except Exception as e:
            logger.warning(f"Could not load {file_path.name}: {e}")

    return docs


def build_vectorstore(reports_dir=REPORTS_DIR, db_dir=CHROMA_DIR, force_rebuild=False):
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(db_dir) and not force_rebuild:
        logger.info(f"Loading existing vector store from {db_dir}")
        return Chroma(persist_directory=db_dir, embedding_function=embeddings)

    docs = load_documents(reports_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Building vector store with {len(chunks)} chunks")

    os.makedirs(db_dir, exist_ok=True)
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=db_dir)
    return vs


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — LLM Config + Core Pipeline
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior climate, energy, and financial risk analyst
embedded within SSE Renewables (UK & Ireland), one of Europe's leading
renewable energy companies.

COMPANY CONTEXT — SSE Renewables:
• Listed on London Stock Exchange (SSE.L), FTSE 100 component
• Owns and operates wind, hydro and solar assets across UK and Ireland
• Flagship assets: Dogger Bank (world's largest offshore wind farm, 3.6GW),
  Seagreen (1.075GW), Viking (443MW onshore), Gordonbush, Bhlaraidh
• Net zero target: 2050 (80% reduction by 2030 vs 2018 baseline)
• Committed £18bn capital investment in low-carbon over 5 years
• Revenue streams: Contracts for Difference (CfD), Renewable Obligation
  Certificates (ROC), merchant power, capacity market payments
• Turbine fleet: primarily Siemens Gamesa SG 14-236 DD (offshore),
  Enercon E-126 / Vestas V136 (onshore)
• Regulatory exposure: Ofgem, NESO, UK CCC climate targets

FINANCIAL PARAMETERS (use these for all calculations):
• Nominal turbine capacity    : 4.2 MW (E-126 class, onshore reference)
• Assumed fleet size          : 50 turbines (210 MW total installed capacity)
• CfD strike price            : £98/MWh (2023 AR5 reference)
• Merchant power price        : £85/MWh (UK day-ahead average reference)
• Annual O&M cost             : £120,000 per turbine (£6M fleet total)
• Availability factor         : 97% (industry standard onshore)
• Transmission loss factor    : 2%
• Carbon intensity avoided    : 0.233 tCO2e/MWh (UK grid average)

BDH MODEL RULES:
• Wind speed is extrapolated to 135m hub height via log-law
• Power output is from physical turbine modelling — do NOT recalculate
• memory_norm = BDH latent stability (higher = more predictable conditions)
• All BDH numerical values are authoritative

Your role IS to:
• Write a professional monthly report with general weather/wind summary
• Calculate and present key financial metrics using the parameters above
• Assess physical and transition climate risks aligned with TCFD
• Reference SSE corporate strategy and targets from the RAG documents
• Provide actionable recommendations specific to SSE operations

NEVER invent data, recalculate BDH physics, or speculate beyond provided values.
Always show your financial calculations step by step.
"""

TASK_PROMPTS = {
    "qa": (
        "Answer the question clearly and concisely using the SSE report context "
        "and live BDH wind data provided. Cite specific sources where possible."
    ),
    "risk_analysis": (
        "You are writing the MONTHLY OPERATIONAL REPORT for SSE Renewables.\n"
        "Structure your response EXACTLY as follows:\n\n"
        "## 1. MONTHLY WEATHER & WIND SUMMARY\n"
        "## 2. ENERGY GENERATION ESTIMATE\n"
        "## 3. FINANCIAL PERFORMANCE ESTIMATE\n"
        "## 4. PHYSICAL CLIMATE RISK ASSESSMENT (TCFD)\n"
        "## 5. SSE STRATEGIC ALIGNMENT\n"
        "## 6. RECOMMENDED ACTIONS\n\n"
        "Use professional financial reporting language. Show ALL calculations."
    ),
    "recommendation": (
        "Based on the BDH predictions and SSE report context, "
        "provide 3-5 actionable recommendations. Prioritise by impact and feasibility. "
        "Reference specific SSE strategic targets where relevant."
    ),
    "scenario": (
        "Analyse the climate scenario implications using the BDH data and SSE report context. "
        "Consider 1.5°C, 2°C, and 3°C+ warming pathways. "
        "Focus on wind resource changes, operational risks, and portfolio resilience."
    ),
    "esg": (
        "Provide a structured ESG/TCFD analysis using the SSE report context and live BDH wind data:\n"
        "1. **Governance** — oversight structures for climate risk\n"
        "2. **Strategy** — climate risk integration into business strategy\n"
        "3. **Risk Management** — identification and management processes\n"
        "4. **Metrics & Targets** — KPIs, net zero commitments, progress"
    ),
}

# Fleet / financial constants
TURBINE_CAPACITY_MW = 4.2
FLEET_SIZE          = 50
TOTAL_CAPACITY_MW   = TURBINE_CAPACITY_MW * FLEET_SIZE
CfD_PRICE           = 98.0
MERCHANT_PRICE      = 85.0
OM_ANNUAL_PER_TURB  = 120_000
OM_MONTHLY          = (OM_ANNUAL_PER_TURB * FLEET_SIZE) / 12
AVAILABILITY        = 0.97
CARBON_FACTOR       = 0.233
RATED_WS            = 12.0

# Global live state
LIVE_STATE: dict = {
    "hour":          0,
    "timestamp":     None,
    "features":      {},
    "memory_norm":   None,
    "wind_metrics":  {},
    "recent_errors": [],
}

# Module-level singletons (populated by init_pipeline)
bdh_model:     BDH | None  = None
retriever                  = None
all_monthly_records: list  = []
groq_client:   Groq | None = None
JSON_OUTPUT_PATH = os.environ.get("JSON_OUTPUT_PATH", "/app/sse_monthly_analysis.json")


def compute_financials(avg_ws: float, hours: int, lo_wind: int) -> dict:
    cf        = min(0.45, 0.45 * (avg_ws / RATED_WS) ** 3) * AVAILABILITY
    energy    = round(cf * TOTAL_CAPACITY_MW * hours, 1)
    revenue   = round(energy * CfD_PRICE, 0)
    profit    = round(revenue - OM_MONTHLY, 0)
    lost_rev  = round(lo_wind * TOTAL_CAPACITY_MW * CfD_PRICE, 0)
    carbon    = round(energy * CARBON_FACTOR, 1)
    return {
        "fleet_capacity_mw":    TOTAL_CAPACITY_MW,
        "capacity_factor_pct":  round(cf * 100, 2),
        "est_energy_mwh":       energy,
        "est_revenue_cfd_gbp":  revenue,
        "monthly_om_cost_gbp":  OM_MONTHLY,
        "est_gross_profit_gbp": profit,
        "lost_revenue_gbp":     lost_rev,
        "carbon_avoided_tco2e": carbon,
        "cfd_strike_price_gbp": CfD_PRICE,
    }


def parse_llm_sections(text: str) -> dict:
    section_map = {
        "weather_wind_summary":  r"##\s*1\.",
        "energy_generation":     r"##\s*2\.",
        "financial_performance": r"##\s*3\.",
        "climate_risk_tcfd":     r"##\s*4\.",
        "strategic_alignment":   r"##\s*5\.",
        "recommended_actions":   r"##\s*6\.",
    }
    keys   = list(section_map.keys())
    pats   = list(section_map.values())
    splits = [re.search(p, text) for p in pats]
    sections = {}
    for i, key in enumerate(keys):
        if splits[i] is None:
            sections[key] = ""
            continue
        start = splits[i].start()
        end   = splits[i + 1].start() if (i + 1 < len(keys) and splits[i + 1]) else len(text)
        body  = text[start:end]
        body  = re.sub(r"^##\s*\d+\.\s*[^\n]*\n", "", body, count=1)
        sections[key] = body.strip()
    return sections


def ask_analyst(question: str, task: str = "qa", temperature: float = 0.3,
                chat_history: list | None = None, bdh_data: dict | None = None):
    """Unified analyst: BDH data + RAG + Groq LLM. Returns (answer, sources)."""
    global retriever, groq_client

    if retriever is None or groq_client is None:
        raise RuntimeError("Pipeline not initialised. Call init_pipeline() first.")

    # 1. RAG retrieval
    docs          = retriever.invoke(question)
    context_parts = []
    sources       = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        src  = meta.get("source_file", meta.get("source", "Unknown"))
        page = meta.get("page", "?")
        year = meta.get("report_year", "?")
        context_parts.append(
            f"[Excerpt {i} | {src} | Year: {year} | Page: {page}]\n"
            f"{doc.page_content.strip()}"
        )
        sources.append({"file": src, "year": year, "page": page,
                        "excerpt": doc.page_content[:200]})
    report_context = "\n\n".join(context_parts)

    # 2. Resolve BDH data
    if bdh_data is not None:
        ms        = bdh_data["monthly_summary"]
        fin       = bdh_data["financials"]
        raw_preds = bdh_data.get("raw_predictions", {})
    else:
        ms        = LIVE_STATE.get("wind_metrics", {})
        fin       = {}
        raw_preds = {}

    hours    = int(ms.get("total_hours_processed", 720))
    avg_ws   = float(ms.get("wind_speed_avg_ms",    ms.get("avg_ws_24h",          0)) or 0)
    max_ws   = float(ms.get("wind_speed_max_ms",    0) or 0)
    min_ws   = float(ms.get("wind_speed_min_ms",    0) or 0)
    std_ws   = float(ms.get("wind_speed_std_ms",    0) or 0)
    wp_proxy = float(ms.get("wind_power_proxy_avg", ms.get("avg_power_proxy_24h", 0)) or 0)
    hi_wind  = int(ms.get("high_wind_hours_gt12ms", ms.get("high_wind_hours_24h", 0)) or 0)
    lo_wind  = int(ms.get("low_wind_hours_lt4ms",   ms.get("low_wind_hours_24h",  0)) or 0)
    calm_frac= float(ms.get("calm_fraction_pct",    0) or 0)
    mem_avg  = ms.get("memory_norm_avg",  LIVE_STATE.get("memory_norm", "N/A"))
    mem_std  = ms.get("memory_norm_std",  "N/A")
    pred_err = ms.get("mean_bdh_prediction_error", ms.get("mean_pred_error", "N/A"))
    month_lbl= ms.get("month", LIVE_STATE.get("timestamp", "N/A"))

    if fin:
        cf_pct   = fin["capacity_factor_pct"]
        energy   = fin["est_energy_mwh"]
        revenue  = fin["est_revenue_cfd_gbp"]
        profit   = fin["est_gross_profit_gbp"]
        lost_rev = fin["lost_revenue_gbp"]
        carbon   = fin["carbon_avoided_tco2e"]
    else:
        _f       = compute_financials(avg_ws, hours, lo_wind)
        cf_pct   = _f["capacity_factor_pct"]
        energy   = _f["est_energy_mwh"]
        revenue  = _f["est_revenue_cfd_gbp"]
        profit   = _f["est_gross_profit_gbp"]
        lost_rev = _f["lost_revenue_gbp"]
        carbon   = _f["carbon_avoided_tco2e"]

    bdh_context = f"""
═══════════════════════════════════════════════════════════════
BDH PHYSICS MODEL OUTPUT — {month_lbl}
═══════════════════════════════════════════════════════════════

MODEL PERFORMANCE:
• Hours processed          : {hours}
• Mean prediction error    : {pred_err}
• BDH memory norm (avg)    : {mem_avg}
• BDH memory norm (std)    : {mem_std}

WIND RESOURCE:
• Average wind speed       : {avg_ws:.3f} m/s
• Maximum wind speed       : {max_ws:.3f} m/s
• Minimum wind speed       : {min_ws:.3f} m/s
• Wind speed std deviation : {std_ws:.3f} m/s
• Wind power proxy (avg)   : {wp_proxy:.3f}
• High-wind hours >12 m/s  : {hi_wind}
• Low-wind  hours  <4 m/s  : {lo_wind}
• Calm fraction            : {calm_frac:.1f} %

═══════════════════════════════════════════════════════════════
FLEET PARAMETERS & PRE-COMPUTED FINANCIALS
═══════════════════════════════════════════════════════════════
Fleet size           : {FLEET_SIZE} turbines × {TURBINE_CAPACITY_MW} MW = {TOTAL_CAPACITY_MW:.0f} MW total
CfD strike price     : £{CfD_PRICE}/MWh
Monthly O&M cost     : £{OM_MONTHLY:,.0f}

FINANCIAL OUTPUTS:
• Capacity factor          : {cf_pct:.2f}%
• Est. energy generated    : {energy:,.1f} MWh
• Est. CfD revenue         : £{revenue:,.0f}
• Est. gross profit        : £{profit:,.0f}
• Lost revenue (low-wind)  : £{lost_rev:,.0f}
• Carbon avoided           : {carbon:,.1f} tCO2e
"""

    task_instruction = TASK_PROMPTS.get(task, TASK_PROMPTS["qa"])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({
        "role": "user",
        "content": (
            f"{task_instruction}\n\n"
            f"{'='*60}\nSSE CORPORATE REPORT CONTEXT\n{'='*60}\n"
            f"{report_context}\n\n"
            f"{'='*60}\nLIVE BDH MODEL OUTPUT\n{'='*60}\n"
            f"{bdh_context}\n\n"
            f"QUESTION: {question}"
        ),
    })

    stream = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        stream=True,
    )
    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_response += delta

    return full_response, sources


# ─────────────────────────────────────────────────────────────────────────────
# Startup initialiser — called once when API boots
# ─────────────────────────────────────────────────────────────────────────────

def init_pipeline():
    """Initialise Groq client, RAG vector store, and BDH model weights from HuggingFace."""
    global groq_client, retriever, bdh_model

    # ── Groq ──────────────────────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
    groq_client = Groq(api_key=api_key)
    logger.info("✅ Groq client initialised.")

    # ── RAG vector store ──────────────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    vs        = build_vectorstore(REPORTS_DIR, CHROMA_DIR)
    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    logger.info("✅ RAG retriever ready.")

    # ── BDH model — download from HuggingFace, fallback to local path ─────────
    config    = BDHConfig()
    bdh_model = BDH(config, input_dim=FEATURE_DIM, output_dim=FEATURE_DIM).to(DEVICE)

    model_path = None

    hf_token   = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")

    if hf_repo_id:
        try:
            from huggingface_hub import hf_hub_download
            logger.info(f"⬇️  Downloading BDH weights from HuggingFace: {hf_repo_id}")
            model_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename="bdh_model.pt",
                token=hf_token,
            )
            logger.info("✅ BDH weights downloaded from HuggingFace.")
        except Exception as e:
            logger.warning(f"⚠️  HuggingFace download failed: {e}")

    # Fallback to local path if HF download didn't work
    if not model_path:
        local_path = os.environ.get("BDH_MODEL_PATH", "/app/bdh_model.pt")
        if os.path.exists(local_path):
            model_path = local_path
            logger.info(f"✅ Using local BDH weights: {local_path}")

    if model_path:
        state = torch.load(model_path, map_location=DEVICE)
        bdh_model.load_state_dict(state)
        logger.info("✅ BDH model weights loaded successfully.")
    else:
        logger.warning(
            "⚠️  No BDH weights found (HuggingFace or local). "
            "Using uninitialised model — /ask still works via Groq + RAG."
        )

    bdh_model.eval()
    logger.info("🚀 Pipeline initialisation complete.")
