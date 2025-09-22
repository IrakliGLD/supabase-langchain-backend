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

# Import DB documentation context
from context import DB_SCHEMA_DOC

# LlamaIndex imports
from llama_index.core import SQLDatabase as LI_SQLDatabase
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import NLSQLTableQueryEngine

# Load .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

if not OPENAI_API_KEY or not SUPABASE_DB_URL:
    raise RuntimeError("Missing OPENAI_API_KEY or SUPABASE_DB_URL in environment")

# Supabase (Postgres) DB connection
engine = create_engine(SUPABASE_DB_URL, pool_pre_ping=True)

# Restrict DB access ONLY to documented tables/views
allowed_tables = [
    "energy_balance_long",
    "entities",
    "monthly_cpi",
    "price",
    "tariff_gen",
    "tech_quantity",
    "trade"
]

# Wrap SQLAlchemy engine for LlamaIndex
li_db = LI_SQLDatabase(engine, include_tables=allowed_tables)

# Full system prompt merged with schema docs
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
🧠 THINK ABOUT THE STORY: What is the user trying to understand?
- Trends over time → Line charts show progression and patterns
- Comparisons between categories → Bar charts show relative magnitudes
- Proportional breakdowns → Pie charts show parts of a whole
- Distribution analysis → Consider the number of categories and data density

🧠 CONSIDER DATA CHARACTERISTICS:
- Time series data (monthly, yearly) → Line charts reveal trends
- Few categories (2-6 items) → Pie charts work well for composition
- Many categories (>10 items) → Bar charts prevent overcrowding
- Comparison queries → Bar charts highlight differences

🧠 QUERY CONTEXT CLUES:
- Words like "trend", "over time", "monthly", "progression" → Think time series visualization
- Words like "share", "proportion", "breakdown", "composition" → Think proportional visualization
- Words like "compare", "vs", "between", "against" → Think comparative visualization
- Words like "generation", "consumption" with time periods → Think trend analysis

=== RESPONSE FORMATTING ===
📝 FOR TEXT ANSWERS:
- Provide clear, structured summaries
- Use bullet points or tables for multiple data points
- Include context: time periods, sectors, units of measurement
- Round numbers appropriately
- Highlight key insights or trends

📈 FOR CHART REQUESTS:
- Provide concise description
- Ensure data is returned in structured format (date/category + value)
- Keep explanations minimal when chart is requested

=== SCHEMA DOCUMENTATION ===
{DB_SCHEMA_DOC}

REMEMBER: You are a data analyst with visualization expertise. Stick strictly to the database data!
"""

# Initialize LLM + Query engine
llm = OpenAI(model="gpt-5-mini", api_key=OPENAI_API_KEY, temperature=0)

query_engine = NLSQLTableQueryEngine(
    sql_database=li_db,
    llm=llm,
    context_str=SYSTEM_PROMPT,
    verbose=True,
)

# FastAPI app
app = FastAPI(title="Supabase LlamaIndex Backend")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str

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

def is_chart_request(query: str) -> tuple[bool, str]:
    query_lower = query.lower()
    chart_keywords = ["chart", "plot", "graph", "visualize", "show as", "display as"]
    is_chart = any(keyword in query_lower for keyword in chart_keywords)
    
    if not is_chart:
        return False, None
    if "pie" in query_lower:
        return True, "pie"
    elif "line" in query_lower or "trend" in query_lower:
        return True, "line"
    else:
        return True, "bar"

def process_sql_results_for_chart(raw_results, query: str):
    if not raw_results or not isinstance(raw_results, list):
        return []
    chart_data = []
    for row in raw_results:
        row = convert_decimal_to_float(row)
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        if len(row) == 2:
            col1, col2 = row
            chart_data.append({"x": str(col1), "y": float(col2)})
        elif len(row) == 3:
            col1, col2, col3 = row
            chart_data.append({"x": str(col1), "group": str(col2), "y": float(col3)})
    return chart_data

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
        schema = li_db.get_table_info()
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(q: Question, x_app_key: str = Header(...)):
    if not APP_SECRET_KEY or x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        is_chart, chart_type = is_chart_request(q.query)

        response = query_engine.query(q.query)
        raw_answer = str(response)

        chart_data = []
        final_answer = raw_answer

        if is_chart and hasattr(response, "metadata") and "sql_query" in response.metadata:
            sql_query = response.metadata["sql_query"]
            try:
                with engine.connect() as conn:
                    result_proxy = conn.execute(text(sql_query))
                    raw_results = result_proxy.fetchall()
                    if raw_results:
                        chart_data = process_sql_results_for_chart(raw_results, q.query)
                        final_answer = "Here's your data visualization:"
            except Exception as e:
                print(f"Error executing SQL directly: {e}")

        return {
            "answer": final_answer if not chart_data else "Here's your data visualization:",
            "chartType": chart_type if chart_data else None,
            "data": chart_data if chart_data else None
        }

    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
