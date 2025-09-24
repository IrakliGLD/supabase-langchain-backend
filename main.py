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
✅ CLEAN SQL ONLY: Return plain SQL text without markdown fences.
✅ SCHEMA COMPLIANCE: Use only documented tables/columns.
✅ FLEXIBLE MATCHING: Handle typos gracefully.
✅ PROPER AGGREGATION: Use SUM/AVG/COUNT appropriately.
✅ SMART FILTERING: Apply WHERE clauses for date ranges, sectors, sources.
✅ LOGICAL JOINS: Only join when schema relationships clearly support it.
✅ PERFORMANCE AWARE: Use LIMIT for large datasets, but NEVER when analyzing trends.

=== DATA PRESENTATION INTELLIGENCE ===
- Trends → Line charts
- Comparisons → Bar charts
- Proportions → Pie charts
- Many categories → Bars to avoid clutter
- Always order time series chronologically

=== TREND & ANALYSIS RULES ===
- If "trend" requested without period → default to 2015–2025 full dataset.
- Always use ALL available data for trends, not just samples.
- Mention SEASONALITY when analyzing generation:
  • Hydro ↑ spring/summer, ↓ winter
  • Thermal ↑ winter, or when hydro is low
  • Imports/exports → seasonal variation possible
- Never analyze just a few rows unless explicitly requested.
- Prefer monthly/yearly aggregation unless daily requested.

=== RESPONSE FORMATTING ===
📝 TEXT:
- Clear summaries with context (time, units).
- Round numbers (e.g., 1,083.9 not 1083.87439).
- Highlight key insights: trends, peaks, changes.

📈 CHARTS:
- Return structured data for plotting (time series: date+value).
- Explanations minimal when charts requested.

=== ERROR HANDLING ===
❌ No data → "I don't have data for that request."
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
        return [convert_decimal_to_float(i) for i in obj]
    if isinstance(obj, tuple):
        return tuple(convert_decimal_to_float(i) for i in obj)
    if isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    return obj

def format_number(value: float, unit: str = None) -> str:
    if value is None: return "0"
    try: formatted = f"{float(value):,.1f}"
    except: formatted = str(value)
    return f"{formatted} {unit}" if unit and unit != "Value" else formatted

def detect_unit(query: str):
    q = query.lower()
    if "price" in q or "tariff" in q:
        return "USD/MWh" if "usd" in q else "GEL/MWh"
    if any(w in q for w in ["generation","consume","energy","trade","import","export"]):
        return "TJ"
    return "Value"

def is_chart_request(query: str):
    q = query.lower()
    if "pie" in q: return True,"pie"
    if any(w in q for w in ["line","trend","over time"]): return True,"line"
    if any(w in q for w in ["chart","plot","graph","visual"]): return True,"bar"
    return False,None

def detect_analysis_type(query: str):
    q = query.lower()
    if "trend" in q: return "trend"
    if any(w in q for w in ["forecast","predict","projection"]): return "forecast"
    if "yoy" in q or "year-over-year" in q: return "yoy"
    if "mom" in q or "month-over-month" in q: return "mom"
    if "cagr" in q or "growth rate" in q: return "cagr"
    return "none"

def scrub_schema_mentions(text: str) -> str:
    if not text: return text
    for t in allowed_tables:
        text = re.sub(rf"\b{re.escape(t)}\b","the database",text,flags=re.I)
    return re.sub(r"\b(schema|table|column|sql|join)\b","data",text,flags=re.I)

# ---------- Short-memory (last 3 Q/A pairs) ----------
def get_recent_history(user_id: str, limit_pairs=3):
    if not user_id: return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""SELECT role, content
                        FROM chat_history
                        WHERE user_id=:uid
                        ORDER BY created_at DESC
                        LIMIT :lim"""),
                {"uid":user_id,"lim":limit_pairs*2}
            ).fetchall()
        return [{"role":r[0],"content":r[1]} for r in rows[::-1]]
    except: return []

def build_memory_context(user_id: str|None, query: str) -> str:
    history = get_recent_history(user_id,3) if user_id else []
    if not history: return query
    hist = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
    return f"{hist}\nUser: {query}\nAssistant:"

# ---------- Trend enforcement ----------
def enforce_full_data_for_trend(sql_cmd: str, analysis_type: str) -> str:
    if analysis_type=="trend":
        sql_cmd = re.sub(r"\s+LIMIT\s+\d+","",sql_cmd,flags=re.I)
    return sql_cmd

# ---------- Core shaping ----------
def process_sql_results_for_chart(raw_results, query, unit="Value"):
    chart_data=[]; meta={"title":"Energy Data","xAxisTitle":"Category","yAxisTitle":f"Value ({unit})"}
    df=pd.DataFrame(raw_results,columns=["date_or_cat","value"])
    df["value"]=df["value"].apply(convert_decimal_to_float).astype(float)
    try:
        df["date_or_cat"]=pd.to_datetime(df["date_or_cat"]); df=df.sort_values("date_or_cat")
        for _,r in df.iterrows():
            chart_data.append({"date":str(r["date_or_cat"].date()),"value":round(r["value"],1)})
        meta["xAxisTitle"]="Date"
    except: 
        for r in raw_results:
            chart_data.append({"category":str(r[0]),"value":round(float(convert_decimal_to_float(r[1])),1)})
    return chart_data,meta

# ---------- Endpoints ----------
@app.get("/healthz")
def health():
    try:
        with engine.connect() as c: c.execute(text("SELECT 1"))
        return {"ok":True}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.post("/ask")
def ask(q: Question, x_app_key: str=Header(...)):
    if not APP_SECRET_KEY or x_app_key!=APP_SECRET_KEY:
        raise HTTPException(status_code=401,detail="Unauthorized")
    try:
        is_chart,chart_type=is_chart_request(q.query)
        analysis_type=detect_analysis_type(q.query)
        unit=detect_unit(q.query)
        final_input=build_memory_context(q.user_id,q.query) if q.user_id else q.query

        llm=ChatOpenAI(model="gpt-4o",temperature=0,openai_api_key=OPENAI_API_KEY)
        toolkit=SQLDatabaseToolkit(db=db,llm=llm)
        agent=create_sql_agent(llm=llm,toolkit=toolkit,verbose=True,
                               agent_type="openai-tools",system_message=SYSTEM_PROMPT)

        result=agent.invoke({"input":final_input},return_intermediate_steps=True)
        response_text=result.get("output",""); steps=result.get("intermediate_steps",[])

        if steps:
            last_step=steps[-1]; sql_cmd=None
            if isinstance(last_step,tuple) and "sql_cmd" in last_step[0]: sql_cmd=last_step[0]["sql_cmd"]
            elif isinstance(last_step,dict) and "sql_cmd" in last_step: sql_cmd=last_step["sql_cmd"]

            if sql_cmd:
                sql_cmd=enforce_full_data_for_trend(sql_cmd,analysis_type)
                with engine.connect() as conn:
                    raw=conn.execute(text(clean_sql(sql_cmd))).fetchall()
                if raw:
                    chart_data,meta=process_sql_results_for_chart(raw,q.query,unit)

                    # Autonomous analysis
                    if analysis_type=="trend":
                        analysis_prompt=f"""You are EnerBot. Perform a TREND ANALYSIS.
User query: {q.query}
Units: {unit}
Data: {chart_data}

Explain:
- Overall direction (↑/↓/flat) across full 2015–2025 period
- Seasonal patterns (hydro vs thermal)
- Highs/lows, anomalies
- One-line conclusion
No SQL or schema details."""
                    elif analysis_type=="forecast":
                        analysis_prompt=f"""You are EnerBot. Forecast values from data.
User query: {q.query}
Units: {unit}
Data: {chart_data}

Estimate value for requested future date (e.g., 2030) by simple trend fit.
Explain reasoning clearly, no SQL/schema."""
                    else:
                        analysis_prompt=f"""You are EnerBot. Analyze and summarize this dataset.
User query: {q.query}
Units: {unit}
Data: {chart_data}
Give clear insights (direction, peaks, comparisons)."""

                    analysis_msg=llm.invoke(analysis_prompt)
                    auto_text=getattr(analysis_msg,"content",str(analysis_msg))
                    return {"answer":scrub_schema_mentions(auto_text),
                            "chartType":chart_type if is_chart else None,
                            "data":chart_data if is_chart else None,
                            "chartMetadata":meta if is_chart else None}
        return {"answer":scrub_schema_mentions(response_text)}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
