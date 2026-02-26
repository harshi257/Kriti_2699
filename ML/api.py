"""
api.py — SSE Renewables BDH Climate Risk API  (Render-ready)
=============================================================
Production FastAPI server. Imports all logic from pipeline.py.

HOW TO RUN LOCALLY
------------------
  export GROQ_API_KEY=your_key_here
  uvicorn api:app --host 0.0.0.0 --port 8000

HOW TO DEPLOY ON RENDER
-----------------------
  1. Push this repo to GitHub.
  2. Create a new Web Service on Render.
  3. Build Command : pip install -r requirements.txt
  4. Start Command : uvicorn api:app --host 0.0.0.0 --port $PORT
  5. Add env var   : GROQ_API_KEY = <your key>
  6. (Optional)    : BDH_MODEL_PATH, REPORTS_DIR, CHROMA_DIR, JSON_OUTPUT_PATH

ENDPOINTS
---------
GET  /                          → health check
GET  /monthly-reports           → full JSON output
GET  /monthly-reports/{month}   → single month (e.g. 2023-07)
GET  /live-state                → current LIVE_STATE snapshot
POST /ask                       → query the LLM analyst
POST /upload-report             → upload a PDF/CSV to the RAG corpus
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pipeline as pl

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — runs once at startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up — initialising pipeline...")
    try:
        pl.init_pipeline()
    except Exception as e:
        logger.error(f"Pipeline init failed: {e}")
        # Don't crash the server — let health check report the issue
    yield
    logger.info("🛑 Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SSE Renewables BDH Climate Risk API",
    description=(
        "REST API for the BDH pipeline: query the LLM analyst, "
        "retrieve monthly reports, and inspect live BDH state."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow your deployed frontend to call this API
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "*"  # tighten this to your frontend URL in production, e.g. "https://myapp.vercel.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    task: Literal["qa", "risk_analysis", "recommendation", "scenario", "esg"] = "qa"
    temperature: float = 0.3
    include_sources: bool = True


class AskResponse(BaseModel):
    question: str
    task: str
    answer: str
    sources: list
    live_state_snapshot: dict


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Check API is running and return pipeline status."""
    return {
        "status": "ok",
        "pipeline": "SSE Renewables BDH Climate Risk",
        "monthly_records_loaded": len(pl.all_monthly_records),
        "retriever_ready": pl.retriever is not None,
        "groq_ready": pl.groq_client is not None,
        "live_state_hour": pl.LIVE_STATE.get("hour", "not started"),
        "live_state_timestamp": pl.LIVE_STATE.get("timestamp", "not started"),
    }


@app.get("/monthly-reports", tags=["Reports"])
def get_all_monthly_reports():
    """Return the full 2-year JSON output (all monthly LLM analyses)."""
    json_path = pl.JSON_OUTPUT_PATH
    if os.path.exists(json_path):
        return FileResponse(
            json_path,
            media_type="application/json",
            filename="sse_monthly_analysis.json",
        )
    if not pl.all_monthly_records:
        raise HTTPException(
            status_code=404,
            detail="No monthly records found. Run the pipeline first.",
        )
    return JSONResponse(content=pl.all_monthly_records)


@app.get("/monthly-reports/{month}", tags=["Reports"])
def get_monthly_report(month: str):
    """Return a single month's record. month format: YYYY-MM (e.g. 2023-07)"""
    match = [r for r in pl.all_monthly_records if r.get("month") == month]
    if not match:
        available = [r["month"] for r in pl.all_monthly_records]
        raise HTTPException(
            status_code=404,
            detail=f"Month '{month}' not found. Available: {available}",
        )
    return JSONResponse(content=match[0])


@app.get("/live-state", tags=["BDH"])
def get_live_state():
    """Return the current LIVE_STATE snapshot from the BDH stream."""
    if not pl.LIVE_STATE.get("wind_metrics"):
        return JSONResponse(content={
            "status": "empty",
            "message": (
                "LIVE_STATE has no wind metrics yet. "
                "Upload NASA data and run the pipeline, or /ask will still work "
                "with fallback values."
            ),
            "raw": pl.LIVE_STATE,
        })
    return JSONResponse(content=pl.LIVE_STATE)


@app.post("/ask", response_model=AskResponse, tags=["Analyst"])
def ask_endpoint(req: AskRequest):
    """
    Query the LLM analyst with a custom question.

    The analyst combines:
    - Live BDH physics state (LIVE_STATE)
    - RAG retrieval over SSE corporate documents
    - Groq LLM (llama-3.3-70b-versatile)

    **task options:**
    - `qa`              — general question & answer
    - `risk_analysis`   — structured TCFD monthly report
    - `recommendation`  — 3-5 actionable recommendations
    - `scenario`        — climate scenario analysis (1.5°C / 2°C / 3°C+)
    - `esg`             — ESG / TCFD structured analysis

    **Example curl:**
    ```bash
    curl -X POST https://your-app.onrender.com/ask \\
      -H "Content-Type: application/json" \\
      -d '{"question": "What are the main wind energy risks this month?", "task": "risk_analysis"}'
    ```
    """
    if pl.retriever is None or pl.groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not ready. Check server logs — GROQ_API_KEY may be missing.",
        )
    try:
        answer, sources = pl.ask_analyst(
            question=req.question,
            task=req.task,
            temperature=req.temperature,
        )
        return AskResponse(
            question=req.question,
            task=req.task,
            answer=answer,
            sources=sources if req.include_sources else [],
            live_state_snapshot={
                "hour":         pl.LIVE_STATE.get("hour"),
                "timestamp":    pl.LIVE_STATE.get("timestamp"),
                "memory_norm":  pl.LIVE_STATE.get("memory_norm"),
                "wind_metrics": pl.LIVE_STATE.get("wind_metrics", {}),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ask_analyst failed")
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")


@app.post("/upload-report", tags=["Admin"])
async def upload_report(file: UploadFile = File(...)):
    """
    Upload a PDF or CSV to the RAG corpus and rebuild the vector store.
    Accepted types: application/pdf, text/csv
    """
    allowed = {".pdf", ".csv"}
    suffix  = os.path.splitext(file.filename)[1].lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {suffix} not supported. Use PDF or CSV.")

    os.makedirs(pl.REPORTS_DIR, exist_ok=True)
    dest = os.path.join(pl.REPORTS_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())

    # Rebuild vector store
    try:
        vs = pl.build_vectorstore(pl.REPORTS_DIR, pl.CHROMA_DIR, force_rebuild=True)
        pl.retriever = vs.as_retriever(search_kwargs={"k": pl.RETRIEVAL_TOP_K})
        return {"status": "ok", "message": f"Uploaded {file.filename} and rebuilt RAG index."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG rebuild failed: {str(e)}")
