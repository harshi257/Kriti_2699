"""
SSE Renewables — BDH Climate Risk API
======================================
FastAPI wrapper around ask_analyst() from the BDH pipeline notebook.

HOW TO USE
----------
1.  Run ALL notebook cells (1–6) first to build:
      • bdh_model       — trained BDH model
      • retriever       — RAG vector store
      • LIVE_STATE      — populated by the stream loop
      • all_monthly_records — JSON output from the pipeline

2.  Then run this file in the same Python process (or import it from a
    Colab cell after the pipeline has run):

      # In a Colab cell, after running cells 1-6:
      import nest_asyncio, uvicorn
      nest_asyncio.apply()
      uvicorn.run(app, host="0.0.0.0", port=8000)

    Or from terminal:
      uvicorn api:app --host 0.0.0.0 --port 8000 --reload

3.  Use ngrok (or similar) to expose the Colab port publicly:
      !pip install pyngrok
      from pyngrok import ngrok
      public_url = ngrok.connect(8000)
      print("API URL:", public_url)

ENDPOINTS
---------
GET  /                          → health check
GET  /monthly-reports           → download the full 2-year JSON output
GET  /monthly-reports/{month}   → single month record (e.g. "2023-07")
GET  /live-state                → current BDH LIVE_STATE snapshot
POST /ask                       → query the LLM analyst
"""

# ── Paste this cell AFTER Cell 6 in your notebook ─────────────────────────────

# Install FastAPI if needed (add to Cell 1 pip installs for production):
# !pip install -q fastapi uvicorn nest_asyncio pyngrok

import json
import os
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ── These globals come from the notebook cells above ──────────────────────────
# ask_analyst, LIVE_STATE, all_monthly_records, JSON_OUTPUT_PATH
# They are already in scope if this code runs in the same kernel.

app = FastAPI(
    title="SSE Renewables BDH Climate Risk API",
    description=(
        "REST API for the BDH pipeline: query the LLM analyst, "
        "retrieve monthly reports, and inspect live BDH state."
    ),
    version="1.0.0",
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
    n_months = len(all_monthly_records) if "all_monthly_records" in globals() else 0
    return {
        "status": "ok",
        "pipeline": "SSE Renewables BDH Climate Risk",
        "monthly_records_loaded": n_months,
        "live_state_hour": LIVE_STATE.get("hour", "not started"),
        "live_state_timestamp": LIVE_STATE.get("timestamp", "not started"),
    }


@app.get("/monthly-reports", tags=["Reports"])
def get_all_monthly_reports():
    """
    Return the full 2-year JSON output (all monthly LLM analyses).
    If the file exists on disk, serve it directly; otherwise use in-memory records.
    """
    json_path = globals().get("JSON_OUTPUT_PATH", "/content/sse_monthly_analysis.json")

    if os.path.exists(json_path):
        return FileResponse(
            json_path,
            media_type="application/json",
            filename="sse_monthly_analysis.json",
        )

    records = globals().get("all_monthly_records", [])
    if not records:
        raise HTTPException(
            status_code=404,
            detail="No monthly records found. Run the pipeline (Cell 6) first.",
        )
    return JSONResponse(content=records)


@app.get("/monthly-reports/{month}", tags=["Reports"])
def get_monthly_report(month: str):
    """
    Return a single month's record.
    month format: YYYY-MM  (e.g. 2023-07)
    """
    records = globals().get("all_monthly_records", [])
    match = [r for r in records if r.get("month") == month]
    if not match:
        available = [r["month"] for r in records]
        raise HTTPException(
            status_code=404,
            detail=f"Month '{month}' not found. Available: {available}",
        )
    return JSONResponse(content=match[0])


@app.get("/live-state", tags=["BDH"])
def get_live_state():
    """
    Return the current LIVE_STATE snapshot from the BDH stream.
    Updated every hour during the pipeline run; reflects the last processed hour.
    """
    state = globals().get("LIVE_STATE", {})
    if not state:
        raise HTTPException(
            status_code=503,
            detail="LIVE_STATE is empty. Run the pipeline (Cell 6) first.",
        )
    return JSONResponse(content=state)


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
    curl -X POST http://localhost:8000/ask \\
      -H "Content-Type: application/json" \\
      -d '{
        "question": "What are the main wind energy risks for SSE this month?",
        "task": "risk_analysis",
        "temperature": 0.3
      }'
    ```
    """
    try:
        answer_fn = globals().get("ask_analyst")
        if answer_fn is None:
            raise HTTPException(
                status_code=503,
                detail="ask_analyst() not found. Ensure notebook cells 1-6 have been run.",
            )

        answer, sources = answer_fn(
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
                "hour": LIVE_STATE.get("hour"),
                "timestamp": LIVE_STATE.get("timestamp"),
                "memory_norm": LIVE_STATE.get("memory_norm"),
                "wind_metrics": LIVE_STATE.get("wind_metrics", {}),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Colab launcher — paste this into a NEW notebook cell (Cell 8)
# ─────────────────────────────────────────────────────────────────────────────

COLAB_LAUNCH_SNIPPET = '''
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 8 — Launch the API Server                            ║
# ║  Run AFTER cells 1–6 have completed.                       ║
# ╚══════════════════════════════════════════════════════════════╝

!pip install -q fastapi uvicorn nest_asyncio pyngrok

import nest_asyncio
import uvicorn
import threading
from pyngrok import ngrok

# Make asyncio work inside Colab
nest_asyncio.apply()

# ── Import the API app (assuming api.py is in /content/) ──────────────────────
import sys
sys.path.insert(0, "/content")
from api import app   # <-- imports the FastAPI app defined in api.py

# ── Expose locally on port 8000 ───────────────────────────────────────────────
PORT = 8000

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# ── Create a public URL via ngrok ─────────────────────────────────────────────
public_url = ngrok.connect(PORT)
print(f"\\n✅ API is live!")
print(f"   Local  : http://localhost:{PORT}")
print(f"   Public : {public_url}")
print(f"\\n📖 Docs available at: {public_url}/docs")
print(f"\\n📡 Example query:")
print(f\'\'\'   curl -X POST {public_url}/ask \\\\
     -H "Content-Type: application/json" \\\\
     -d \'{{"question": "What are the wind risks this month?", "task": "qa"}}\'\'\')
'''

if __name__ == "__main__":
    # Local development — run with: python api.py
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
