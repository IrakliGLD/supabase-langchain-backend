import os 
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
import json
import re
from datetime import datetime

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
db = SQLDatabase(engine, include_tables=allowed_tables)

# --- Patch execution to clean accidental markdown fences ---
def clean_sql(query: str) -> str:
    """Remove markdown fences if GPT accidentally adds them."""
    return query.replace("```sql", "").replace("```", "").strip()

old_execute = db._execute
def cleaned_execute(sql: str, *args, **kwargs):
    sql = clean_sql(sql)
    return old_execute(sql, *args, **kwargs)
db._execute = cleaned_execute
# ----------------------------------------------------------

# FastAPI app
app = FastAPI(title="Supabase LangChain Backend")

# CORS settings (open for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced system prompt
SYSTEM_PROMPT = f"""
You are EnerBot, an expert Georgian electricity market data analyst with advanced data visualization intelligence.

=== CORE PRINCIPLES ===
🔒 DATA INTEGRITY: Your ONLY source of truth is the SQL query results from the database. Never use outside knowledge.
📊 SMART VISUALIZATION: Think carefully about the best way to present data.
🚫 NO HALLUCINATION: If unsure, respond: "I don't know based on the available data."

=== SQL QUERY RULES ===
✅ CLEAN SQL ONLY: No markdown fences.
✅ SCHEMA COMPLIANCE: Use only documented tables/columns.
✅ PROPER AGGREGATION: SUM for totals, AVG for averages, COUNT for quantities.
✅ LOGICAL JOINS: Only when schema supports it.

=== SCHEMA DOCUMENTATION ===
{DB_SCHEMA_DOC}
"""

class Question(BaseModel):
    query: str

def convert_decimal_to_float(obj):
    """Recursively convert Decimal objects to float"""
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
    chart_patterns = [
        "chart", "plot", "graph", "visualize", "visualization", 
        "show as", "display as", "present as", "render as",
        "bar chart", "line chart", "pie chart"
    ]
    is_chart = any(pattern in query_lower for pattern in chart_patterns)
    if not is_chart:
        return False, None
    if any(word in query_lower for word in ["pie", "pie chart"]):
        chart_type = "pie"
    elif any(word in query_lower for word in ["line", "line chart", "trend", "over time"]):
        chart_type = "line"
    else:
        chart_type = "bar"
    return True, chart_type

def intelligent_chart_type_selection(raw_results, query: str, explicit_type: str = None):
    if explicit_type and explicit_type != "auto":
        return explicit_type
    if not raw_results or len(raw_results) == 0:
        return "bar"
    query_lower = query.lower()
    num_rows = len(raw_results)
    sample_row = raw_results[0] if raw_results else []
    if any(word in query_lower for word in ["trend", "over time", "monthly", "yearly"]):
        return "line"
    if any(word in query_lower for word in ["share", "proportion", "breakdown"]):
        return "pie"
    if len(sample_row) >= 2:
        first_col = sample_row[0]
        if isinstance(first_col, str) and ('-' in first_col or '/' in first_col):
            try:
                datetime.strptime(str(first_col)[:10], '%Y-%m-%d')
                return "line"
            except:
                pass
    if num_rows <= 6:
        return "pie"
    return "bar"

def process_sql_results_for_chart(raw_results, query: str):
    chart_data = []
    metadata = {
        "title": "Energy Data Visualization",
        "xAxisTitle": "Category",
        "yAxisTitle": "Value",
        "datasetLabel": "Data"
    }
    for row in raw_results:
        row = convert_decimal_to_float(row)
        if len(row) >= 2:
            col1, col2 = row[0], row[1]
            if isinstance(col1, str):
                try:
                    datetime.strptime(col1, '%Y-%m-%d')
                    chart_item = {"date": str(col1), "value": round(float(col2), 2) if col2 else 0.0}
                    chart_data.append(chart_item)
                    metadata["xAxisTitle"] = "Date"
                except:
                    chart_item = {"sector": str(col1), "volume_tj": round(float(col2), 2) if col2 else 0.0}
                    chart_data.append(chart_item)
                    metadata["xAxisTitle"] = "Category"
    return chart_data, metadata

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
        is_chart, chart_type = is_chart_request(q.query)

        # Direct SQL test case for hydro generation
        if is_chart and "hydro" in q.query.lower() and "generation" in q.query.lower():
            test_sql = """
            SELECT 
                DATE_TRUNC('month', period)::date as month,
                SUM(volume_tj) as generation_mwh
            FROM energy_balance_long 
            WHERE entity_name ILIKE '%hydro%' 
            GROUP BY DATE_TRUNC('month', period)
            ORDER BY month
            """
            with engine.connect() as conn:
                result = conn.execute(text(test_sql))
                raw_results = result.fetchall()
                if raw_results:
                    chart_data, chart_metadata = process_sql_results_for_chart(raw_results, q.query)
                    optimal_chart_type = intelligent_chart_type_selection(raw_results, q.query, chart_type)
                    return {
                        "answer": "Here's your hydro generation data:",
                        "chartType": optimal_chart_type,
                        "data": chart_data,
                        "chartMetadata": chart_metadata
                    }

        # Fallback to LangChain SQL Agent
        llm = ChatOpenAI(
            model="gpt-4o",  # stronger model
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="openai-tools",  # tool-calling agent for structured queries
            system_message=SYSTEM_PROMPT
        )

        response = agent.run(q.query)

        return {
            "answer": response,
            "chartType": None,  # SQLAgent doesn’t return structured chart data automatically
            "data": None,
            "chartMetadata": None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
