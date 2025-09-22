import os
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from decimal import Decimal
import json
import re
from datetime import datetime

# ===== LlamaIndex imports =====
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI as LI_OpenAI
from llama_index.core.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.core.indices.struct_store import SQLDatabase as LI_SQLDatabase

# Import DB documentation context (no circular import)
from context import DB_SCHEMA_DOC

# Load .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

if not OPENAI_API_KEY or not SUPABASE_DB_URL:
    raise RuntimeError("Missing OPENAI_API_KEY or SUPABASE_DB_URL in environment")

# --- SQLAlchemy engine (Supabase Postgres) ---
engine = create_engine(SUPABASE_DB_URL, pool_pre_ping=True)

# Restrict DB access ONLY to documented tables/views
ALLOWED_TABLES = [
    "energy_balance_long",
    "entities",
    "monthly_cpi",
    "price",
    "tariff_gen",
    "tech_quantity",
    "trade",
]

# --- FastAPI app ---
app = FastAPI(title="Supabase LlamaIndex Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# System Prompt (same style you used)
# =========================
SYSTEM_PROMPT = f"""
You are EnerBot, an expert Georgian electricity market data analyst with advanced data visualization intelligence.

=== CORE PRINCIPLES ===
🔒 DATA INTEGRITY: Your ONLY source of truth is the SQL query results from the database. Never use outside knowledge, assumptions, or estimates.
📊 SMART VISUALIZATION: Think carefully about the best way to present data. Consider the nature of the data and user's analytical needs.
🎯 RELEVANT RESULTS: Focus on what the user actually asked for. Don't provide tangential information.
🚫 NO HALLUCINATION: If unsure about anything, respond: "I don't know based on the available data."

=== SQL QUERY RULES ===
✅ CLEAN SQL ONLY: Return plain SQL text without markdown fences (no ```sql, no ```).
✅ SCHEMA COMPLIANCE: Use only documented tables/columns. Double-check all names against the schema.
✅ FLEXIBLE MATCHING: Handle user typos gracefully (e.g., "residencial" → "residential", "elektric" → "electricity").
✅ PROPER AGGREGATION: Use correct SQL functions (SUM for totals, AVG for averages, COUNT for quantities).
✅ SMART FILTERING: Apply appropriate WHERE clauses for date ranges, sectors, and energy sources.
✅ LOGICAL JOINS: Only join tables when schema relationships clearly support it.
✅ PERFORMANCE AWARE: Use LIMIT clauses for large datasets, especially for charts.

=== DATA PRESENTATION INTELLIGENCE ===
(omitted for brevity — same as your prompt)

=== SCHEMA DOCUMENTATION ===
{DB_SCHEMA_DOC}
"""

# =========================
# Utilities
# =========================
class Question(BaseModel):
    query: str

def convert_decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, list):
        return [convert_decimal_to_float(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(convert_decimal_to_float(x) for x in obj)
    if isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    return obj

def is_chart_request(query: str) -> tuple[bool, str]:
    q = query.lower()
    chart_keywords = ["chart", "plot", "graph", "visualize", "visualization", "show as", "display as", "draw", "render as"]
    if not any(k in q for k in chart_keywords):
        return False, None
    if "pie" in q:
        return True, "pie"
    if "line" in q or "trend" in q:
        return True, "line"
    return True, "bar"

def process_sql_results_for_chart(raw_results, query: str):
    """Turn DB rows into chart-friendly dicts."""
    if not raw_results or not isinstance(raw_results, list):
        return [], {}
    chart_data = []
    metadata = {
        "title": "Energy Data",
        "xAxisTitle": "Category",
        "yAxisTitle": "Value",
        "datasetLabel": "Data",
    }
    q = query.lower()
    if any(w in q for w in ["volume", "consumption", "energy"]):
        metadata["yAxisTitle"] = "Volume (TJ)" if ("tj" in q or "terajoule" in q) else "Energy (MWh)"
    elif any(w in q for w in ["generation", "generated", "produce", "output"]):
        metadata["yAxisTitle"] = "Generation (MWh)" if ("mwh" in q or "megawatt" in q) else "Generation (TJ)"
    elif any(w in q for w in ["price", "cost", "tariff"]):
        metadata["yAxisTitle"] = "Price"
    elif any(w in q for w in ["count", "number", "quantity"]):
        metadata["yAxisTitle"] = "Count"
    elif any(w in q for w in ["percentage", "percent", "share"]):
        metadata["yAxisTitle"] = "Percentage (%)"

    for row in raw_results:
        row = convert_decimal_to_float(row)
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        if len(row) == 2:
            c1, c2 = row
            is_date = False
            if isinstance(c1, str):
                for fmt in ['%Y-%m-%d', '%Y-%m', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        datetime.strptime(c1, fmt)
                        is_date = True
                        break
                    except ValueError:
                        continue
            if is_date:
                chart_data.append({"date": str(c1), "value": float(c2)})
                metadata["xAxisTitle"] = "Date"
                metadata["datasetLabel"] = "Value over Time"
            else:
                chart_data.append({"sector": str(c1), "volume_tj": float(c2)})
                metadata["xAxisTitle"] = "Sector" if "sector" in q else "Category"
        elif len(row) == 3:
            c1, c2, c3 = row
            if isinstance(c1, str) and ('-' in c1 or '/' in c1):
                chart_data.append({"date": str(c1), "sector": str(c2), "value": float(c3)})
                metadata["xAxisTitle"] = "Date"
            else:
                chart_data.append({"sector": str(c2), "volume_tj": float(c3)})
                metadata["xAxisTitle"] = "Sector"
        else:
            # assume (period, sector, source, value)
            period, sector, source, value = row[:4]
            chart_data.append({
                "date": str(period),
                "sector": str(sector),
                "energy_source": str(source),
                "volume_tj": float(value),
            })
            metadata["xAxisTitle"] = "Date"

    return chart_data, metadata

def extract_sql_from_li_response(resp) -> str | None:
    """Try several places to get the SQL text from LlamaIndex response."""
    # 1) Standard: resp.metadata.get('sql_query')
    try:
        md = getattr(resp, "metadata", None) or {}
        if isinstance(md, dict) and "sql_query" in md and md["sql_query"]:
            return md["sql_query"]
    except Exception:
        pass

    # 2) Sometimes present under extra_info
    try:
        extra = getattr(resp, "extra_info", None) or {}
        if isinstance(extra, dict) and "sql_query" in extra and extra["sql_query"]:
            return extra["sql_query"]
    except Exception:
        pass

    # 3) Fallback: regex from textual response (if model echoed SQL)
    try:
        txt = str(getattr(resp, "response", "")) or str(resp)
        m = re.search(r"```sql\s*(.*?)\s*```", txt, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"SELECT\s+.*?;", txt, re.DOTALL | re.IGNORECASE)
        if m2:
            return m2.group(0).strip()
    except Exception:
        pass
    return None

# =========================
# LlamaIndex global setup
# =========================
Settings.llm = LI_OpenAI(
    model="gpt-5-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
    timeout=60,
    # Most recent LlamaIndex supports this arg; if your version doesn't,
    # we'll prepend SYSTEM_PROMPT to the query string below as a fallback.
    system_prompt=SYSTEM_PROMPT,
)

# Build LI SQLDatabase and QueryEngine once
li_sql_db = LI_SQLDatabase(engine, include_tables=ALLOWED_TABLES)
# synthesize_response=False helps us get raw rows/SQL; verbose for dev
li_query_engine = NLSQLTableQueryEngine(
    sql_database=li_sql_db,
    tables=ALLOWED_TABLES,
    synthesize_response=False,
    verbose=True,
)

# =========================
# Health + Introspect
# =========================
@app.get("/healthz")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/introspect")
def introspect():
    try:
        # LlamaIndex doesn't expose a schema dump like LangChain, so pull from SQLAlchemy
        with engine.connect() as conn:
            # very light ping; the UI already has DB_SCHEMA_DOC
            pass
        return {"schema_doc_excerpt": DB_SCHEMA_DOC[:1200] + "..."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# Main endpoint
# =========================
@app.post("/ask")
def ask(q: Question, x_app_key: str = Header(...)):
    # 🔒 API Key Authentication
    if not APP_SECRET_KEY or x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Detect chart intent
        is_chart, chart_type = is_chart_request(q.query)

        # Prepend system prompt if your LI version doesn't support system_prompt
        user_query = q.query
        if not getattr(Settings.llm, "system_prompt", None):
            user_query = SYSTEM_PROMPT + "\n\n" + q.query

        # Ask LlamaIndex to generate SQL
        li_resp = li_query_engine.query(user_query)
        final_text_answer = str(getattr(li_resp, "response", "")) or str(li_resp)

        # Try to fetch the SQL it planned to run
        sql_query = extract_sql_from_li_response(li_resp)

        chart_data = []
        chart_meta = {}
        if is_chart and sql_query:
            try:
                with engine.connect() as conn:
                    result_proxy = conn.execute(text(sql_query))
                    raw_results = result_proxy.fetchall()
                if raw_results:
                    chart_data, chart_meta = process_sql_results_for_chart(raw_results, q.query)
                    final_text_answer = "Here's your data visualization:"
                else:
                    final_text_answer = "I don't have data for that specific request in the selected period."
            except Exception as ex:
                # If SQL fails for any reason, keep a clean message instead of debug dump
                print(f"Direct SQL execution error: {ex}")
                final_text_answer = "I couldn't retrieve the data for that request."

        response = {
            "answer": final_text_answer if not chart_data else "Here's your data visualization:",
            "chartType": chart_type if chart_data else None,
            "data": chart_data if chart_data else None,
            "chartMetadata": chart_meta if chart_data else None,
            # Optional: expose the SQL for debugging in your console (not to UI)
            # "debug_sql": sql_query,
        }
        print(f"Final response: {response}")
        return response

    except Exception as e:
        print(f"Error in /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))
