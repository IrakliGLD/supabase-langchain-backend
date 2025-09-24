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

# Restrict DB access to documented tables/views
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
You are EnerBot, an expert Georgian electricity market data analyst with advanced visualization and analytics.

=== CORE PRINCIPLES ===
- DATA INTEGRITY: Only use database results. Never invent data.
- AUTONOMOUS ANALYSIS: Always analyze results beyond listing numbers. Identify trends, changes, comparisons, anomalies, and seasonality automatically, even if not explicitly asked.
- USER-FOCUSED LANGUAGE: Present insights in natural business/energy language, never technical schema terms.

=== SECURITY RULES ===
- NEVER mention table names, column names, SQL, or schema in user-facing output.
- Translate technical names into plain terms:
  • p_bal_gel → "balancing electricity price (GEL)"
  • tariff_gel → "tariff (GEL)"
  • quantity_tech → "technology quantity"
- If asked about structure: reply "I can analyze and explain energy, price, tariff, technology, trade, and index data, covering 2015–2025."

=== SQL RULES ===
- Generate clean SQL only.
- Use only documented tables/columns internally.
- Never output SQL.

=== PRESENTATION RULES ===
- Trends → Line charts
- Comparisons → Bar charts
- Shares → Pie charts
- Always summarize: overall direction, % change, peaks/lows, anomalies, seasonality.
- Highlight actionable insights.

=== ERROR HANDLING ===
- No data → say "I don't have data for that request."
- Ambiguous → ask clarification.

=== SCHEMA DOCUMENTATION (INTERNAL USE ONLY) ===
{DB_SCHEMA_DOC}
"""

class Question(BaseModel):
    query: str
    user_id: str | None = None

# ---------- Helpers ----------
def convert_decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_decimal_to_float(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_decimal_to_float(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    return obj

def scrub_schema_mentions(text: str) -> str:
    """Scrub schema/table/column references before sending to user."""
    if not text:
        return text
    for t in allowed_tables:
        text = re.sub(rf"\b{re.escape(t)}\b", "the database", text, flags=re.I)
    # Replace snake_case tokens that look like columns
    text = re.sub(r"\b[a-z_]{2,}\b", "the data", text, flags=re.I)
    return text

def get_recent_history(user_id: str, limit_pairs: int = 3):
    if not user_id:
        return []
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
                {"uid": user_id, "lim": limit_pairs * 2}
            ).fetchall()
        rows = rows[::-1]
        return [{"role": r[0], "content": r[1]} for r in rows]
    except Exception:
        return []

def build_memory_context(user_id: str | None, query: str) -> str:
    history = get_recent_history(user_id, limit_pairs=3) if user_id else []
    if not history:
        return query
    history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
    return f"{history_text}\nUser: {query}\nAssistant:"

def process_sql_results_for_chart(raw_results, query: str, unit: str = "Value"):
    chart_data = []
    metadata = {
        "title": "Energy Data Visualization",
        "xAxisTitle": "Category",
        "yAxisTitle": f"Value ({unit})",
        "datasetLabel": "Data"
    }
    df = pd.DataFrame(raw_results, columns=["date_or_cat","value"])
    df["value"] = df["value"].apply(convert_decimal_to_float).astype(float)
    try:
        df["date_or_cat"] = pd.to_datetime(df["date_or_cat"])
        df = df.sort_values("date_or_cat")
        for _, r in df.iterrows():
            chart_data.append({"date": str(r["date_or_cat"].date()), "value": round(r["value"], 1)})
        metadata["xAxisTitle"] = "Date"
    except:
        for r in raw_results:
            chart_data.append({"category": str(r[0]), "value": round(float(convert_decimal_to_float(r[1])), 1)})
        metadata["xAxisTitle"] = "Category"
    return chart_data, metadata

# ---------- Endpoint ----------
@app.post("/ask")
def ask(q: Question, x_app_key: str = Header(...)):
    if not APP_SECRET_KEY or x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        final_input = build_memory_context(q.user_id, q.query) if q.user_id else q.query
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True,
                                 agent_type="openai-tools", system_message=SYSTEM_PROMPT)

        result = agent.invoke({"input": final_input}, return_intermediate_steps=True)
        response_text = result.get("output", "")
        steps = result.get("intermediate_steps", [])

        if steps:
            try:
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
You are EnerBot. Analyze the following dataset and answer the question.
User query: {q.query}
Data: {chart_data}

Rules:
- Always analyze the FULL dataset (not just first few rows).
- Identify direction (increase/decrease/flat) and % change from first to last.
- Mention peaks, lows, anomalies, seasonality if relevant.
- If forecast implied, extend the trend with an estimate.
- Never mention tables, columns, or schema. Use only natural business terms.
"""
                            analysis_msg = llm.invoke(analysis_prompt)
                            auto_text = getattr(analysis_msg, "content", str(analysis_msg))
                            final_ans = scrub_schema_mentions(auto_text)
                            return {
                                "answer": final_ans,
                                "chartType": "line" if "date" in chart_data[0] else "bar",
                                "data": chart_data,
                                "chartMetadata": meta
                            }
            except Exception as e:
                safe_text = scrub_schema_mentions(response_text)
                return {"answer": safe_text, "chartType": None, "data": None, "chartMetadata": None}

        safe_text = scrub_schema_mentions(response_text)
        return {"answer": safe_text, "chartType": None, "data": None, "chartMetadata": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
