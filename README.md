# Kriti_2699
# SSE Renewables — Climate Risk Platform

AI-powered wind energy analytics platform built on BDH physics model + Groq LLM + RAG over SSE corporate documents.

1. BDH Physics Model — a custom deep learning model trained on 8 years of NASA hourly weather data (84 features including wind speed, temperature, solar irradiance, pressure, humidity). Every hour it takes the latest weather window and predicts what comes next — giving us predicted vs actual values for every variable in real time.
2. Groq LLM (llama-3.3-70b) — at the end of every month, the BDH model's outputs are packaged and sent to a large language model which writes a full professional risk report. The report covers wind conditions, energy generation estimates, financial performance (CfD revenue, gross profit, carbon avoided), TCFD climate risk assessment, strategic alignment, and recommended actions — all grounded in real BDH numbers, never invented.
3. RAG over SSE Documents — the LLM is connected to a vector database of SSE's own annual reports, TCFD disclosures, and ESG documents. So when it writes a report or answers a question, it references SSE's actual corporate commitments and targets, not generic knowledge.
The result is a platform that streams live model data, stores 2 years of AI-written monthly reports, and lets anyone ask questions to an analyst that knows both the live wind data and SSE's corporate strategy.
---

## What This Does

- **Live Stream** — BDH model predictions vs actuals across 84 weather features, updated every hour
- **Monthly Reports** — LLM-generated 6-section risk analysis for every month, grounded in real BDH output
- **AI Analyst** — Chat interface to query the LLM about wind performance, financials, and climate risk

---

## Folder Structure

```
sse-renewables-platform/
├── backend/
│   ├── FinalBDHPipeline_WithAPI.ipynb   ← ML model + pipeline + LLM
│   └── api.py                            ← FastAPI server (bridge between notebook and frontend)
└── frontend/
    └── sse_dashboard.jsx                 ← React app (3 pages)
```

---

## How to Run

### Backend (run in Google Colab)

1. Open `Copy_of_FinalBDHPipeline_WithAPI.ipynb` in Google Colab
2. Run cells in order: **Cell 1 → Cell 7**
3. When prompted, enter your `GROQ_API_KEY`
4. Upload SSE PDF reports when Cell 5 asks
5. Upload `api.py` to `/content/` via Colab file panel (📁 left sidebar)
6. Cell 7 will print a public URL like:
   ```
   🌍 Public API URL: https://abc123.ngrok.io
   ```
7. **Copy that URL** — frontend needs it

> ⚠️ Requires a free ngrok account. Run `!ngrok authtoken YOUR_TOKEN` before Cell 7.

---

### Frontend

1. Open `sse_dashboard.jsx`
2. Replace the API URL at the top with the ngrok URL from the backend:
   ```javascript
   const API = "https://abc123.ngrok.io"
   ```
3. Run the React app

---

## API Endpoints

| Method | Endpoint | Used By |
|---|---|---|
| `GET` | `/live-state` | Page 1 — Live Stream |
| `GET` | `/monthly-reports` | Page 2 — Monthly Reports |
| `POST` | `/ask` | Page 3 — AI Analyst Chat |

---
---

## Team Members and Contributions

| Name | Contribution |
|------|-------------|
| **[Eashita Karmakar]** | Implemented BDH model architecture and core PyTorch components. |
| **[Harshita Garg]** | Developed training loop, inference pipeline, and model evaluation. |
| **[Manya]** | Built 84-feature climate preprocessing and feature engineering pipeline (NASA POWER data). |
| **[Kaavya Tawade]** | Designed RAG system, LLM prompting logic, and implemented ask_analyst() integration. |
| **[Anjali]** | Implemented streaming pipeline, monthly aggregation, JSON logging, and FastAPI backend (api.py). |
| **[Chhavi]** | Web demo,deployment |
| **[Sharvani]** |Provided technical mentorship on BDH architecture, climate-risk modeling, and overall system design. |
| **[Sandhya]** | Supervised report preparation, presentation structure, and final system evaluation. |



---


---

## Important

- Keep Colab **open and running** during the demo — if it disconnects the API goes offline
- ngrok URL **changes every restart** — update the frontend URL if you restart Colab
- `/ask` endpoint takes 5–15 seconds to respond (LLM call) — this is normal
