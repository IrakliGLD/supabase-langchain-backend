import os
import re
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd

# Import DB documentation context
from context import DB_SCHEMA_DOC

# Load .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

if not OPENAI_API_KEY or not SUPABASE_DB_URL:
    raise RuntimeError("Missing OPENAI_API_KEY or SUPABASE_DB_URL in environment")

# DB connection
engine = create_engine(SUPABASE_DB_URL, pool_pre_ping=True)

# Restrict DB access to documented tables/views (for the LLM tool use)
allowed_tables = [
    "energy_balance_long",
    "entities",
    "monthly_cpi",
    "price",
    "tariff_gen",
    "tech_quantity",
    "trade"
]
db = SQLDatabase(engine, include_tables=allowed_tables)

# --- Patch SQL execution (strip accidental code fences) ---
def clean_sql(query: str) -> str:
    return query.replace("```sql", "").replace("```", "").strip()

old_execute = db._execute
def cleaned_execute(sql: str, *args, **kwargs):
    sql = clean_sql(sql)
    return old_execute(sql, *args, **kwargs)
db._execute = cleaned_execute

# FastAPI app
app = FastAPI(title="Supabase LangChain Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = f"""
You are EnerBot, an expert Georgian electricity market data analyst with advanced visualization and lightweight analytics.

=== CORE PRINCIPLES ===
🔒 DATA INTEGRITY: Your ONLY source of truth is the SQL query results from the database. Never use outside knowledge, assumptions, or estimates.
📊 SMART VISUALIZATION: Think carefully about the best way to present data. Consider the nature of the data and the user's analytical needs.
🎯 RELEVANT RESULTS: Focus on what the user actually asked for. Don't provide tangential information.
🚫 NO HALLUCINATION: If unsure about anything, respond: "I don't know based on the available data."

=== IMPORTANT SECURITY RULES ===
- NEVER reveal table names, column names, or any database structure to the user.
- Use the schema documentation only internally to generate correct SQL.
- If the user asks about database structure or who you are, reply:
  "I’m an electricity market assistant. I can help analyze and explain energy, price, and trade data."
- Always answer in plain language, not SQL. Do not include SQL in your responses.

=== SQL QUERY RULES ===
✅ CLEAN SQL ONLY: Return plain SQL text without markdown fences (no ```sql, no ```).
✅ SCHEMA COMPLIANCE: Use only documented tables/columns.
✅ FLEXIBLE MATCHING: Handle user typos gracefully.
✅ PROPER AGGREGATION: Use SUM/AVG/COUNT appropriately.
✅ SMART FILTERING: Apply WHERE clauses for date ranges, sectors, sources.
✅ PERFORMANCE AWARE: Use LIMIT for large datasets.

=== DATA PRESENTATION INTELLIGENCE ===
- Trends over time → Line charts
- Comparisons → Bar charts
- Proportions → Pie charts
- Time series → Order by date

=== TREND & ANALYSIS RULES ===
- If user requests a "trend" but no period:
  → Default to full available 2015–2025 period (unless clarified).
- Always mention SEASONALITY when analyzing generation:
  • Hydro → higher spring/summer, lower winter
  • Thermal → higher in winter or hydro shortages
- Prefer monthly/yearly aggregation unless daily explicitly asked.

=== AUTONOMOUS ANALYST ROLE ===
- You are free to decide what analysis is most relevant: trend, correlation, seasonality, comparisons, ratios, forecasts, anomalies, rankings, shares, etc.
- Always ground answers ONLY in SQL results from allowed data.
- If vague, interpret reasonably and provide the most useful analysis.
- Present insights in clear, natural language: summarize patterns, explain key takeaways, highlight what matters.
- Do not just list raw numbers unless explicitly asked. Always contextualize.
- Never reveal SQL queries, table names, or schema.

=== RESPONSE FORMATTING ===
📝 TEXT:
- Clear summaries, context, units.
- Round numbers (e.g., 1,083.9 not 1083.87439).
- Highlight insights (trends, peaks, anomalies).

📈 CHARTS:
- If requested, return structured chart_data (time series or categories).
- Keep explanation concise alongside chart.

=== ERROR HANDLING ===
❌ No data → "I don't have data for that specific request."
❌ Ambiguous → Ask for clarification.
❌ Invalid → Suggest alternatives.

=== SCHEMA DOCUMENTATION (FOR INTERNAL USE ONLY) ===
{DB_SCHEMA_DOC}
"""

# Accept optional user_id for memory
class Question(BaseModel):
    query: str
    user_id: str | None = None

# ---------- Helpers ----------
def convert_decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, list):
        return [convert_decimal_to_float(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(convert_decimal_to_float(x) for x in obj)
    if isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k,v in obj.items()}
    return obj

def scrub_schema_mentions(text: str) -> str:
    if not text:
        return text
    for t in allowed_tables:
        text = re.sub(rf"\b{re.escape(t)}\b", "the database", text, flags=re.IGNORECASE)
    return re.sub(r"\b(schema|table|column|sql|join)\b", "data", text, flags=re.IGNORECASE)

# Memory helpers
def get_recent_history(user_id: str, limit_pairs: int = 3):
    if not user_id: return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                SELECT role, content
                FROM chat_history
                WHERE user_id = :uid
                ORDER BY created_at DESC
                LIMIT :lim
                """),
                {"uid": user_id, "lim": limit_pairs*2}
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows[::-1]]
    except Exception:
        return []

def build_memory_context(user_id: str | None, query: str) -> str:
    history = get_recent_history(user_id, 3)
    if not history:
        return query
    htxt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
    return f"{htxt}\nUser: {query}\nAssistant:"

# Chart shaping
def process_sql_results_for_chart(raw_results, query: str, unit="Value"):
    chart_data, meta = [], {
        "title": "Energy Data",
        "xAxisTitle": "Category",
        "yAxisTitle": f"Value ({unit})",
        "datasetLabel": "Data"
    }
    df = pd.DataFrame(raw_results, columns=["date_or_cat","value"])
    df["value"] = df["value"].apply(convert_decimal_to_float).astype(float)
    try:
        df["date_or_cat"] = pd.to_datetime(df["date_or_cat"])
        df = df.sort_values("date_or_cat")
        for _,r in df.iterrows():
            chart_data.append({"date": str(r["date_or_cat"].date()), "value": round(r["value"],1)})
        meta["xAxisTitle"] = "Date"
    except:
        for r in raw_results:
            chart_data.append({"sector": str(r[0]), "value": round(float(convert_decimal_to_float(r[1])),1)})
    return chart_data, meta

# ---------- Endpoints ----------
@app.get("/healthz")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(q: Question, x_app_key: str = Header(...)):
    if not APP_SECRET_KEY or x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        final_input = build_memory_context(q.user_id, q.query) if q.user_id else q.query
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True, agent_type="openai-tools", system_message=SYSTEM_PROMPT)

        result = agent.invoke({"input": final_input}, return_intermediate_steps=True)
        response_text = result.get("output","")
        steps = result.get("intermediate_steps",[])

        if steps:
            last_step = steps[-1]
            sql_cmd = None
            if isinstance(last_step, tuple) and isinstance(last_step[0], dict) and "sql_cmd" in last_step[0]:
                sql_cmd = last_step[0]["sql_cmd"]
            elif isinstance(last_step, dict) and "sql_cmd" in last_step:
                sql_cmd = last_step["sql_cmd"]

            if sql_cmd:
                with engine.connect() as conn:
                    raw = conn.execute(text(clean_sql(sql_cmd))).fetchall()
                if raw:
                    chart_data, meta = process_sql_results_for_chart(raw, q.query)
                    analysis_prompt = f"""
You are EnerBot. Analyze the data below and answer the user's question.
User query: {q.query}
Data: {chart_data}

Write a clear analytical narrative:
- Summarize main patterns (up/down/flat, approximate change).
- Note peaks, lows, anomalies, seasonality if relevant.
- Provide a one-line takeaway.
- Never mention tables, SQL, or schema.
"""
                    analysis_msg = llm.invoke(analysis_prompt)
                    auto_text = getattr(analysis_msg,"content",str(analysis_msg))
                    return {
                        "answer": scrub_schema_mentions(auto_text.strip() or response_text),
                        "chartType": "line",
                        "data": chart_data,
                        "chartMetadata": meta
                    }

        return {"answer": scrub_schema_mentions(response_text), "chartType": None, "data": None, "chartMetadata": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
