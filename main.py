import os
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
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

# --- Patch for GPT-5-mini unsupported "stop" ---
class PatchedChatOpenAI(ChatOpenAI):
    def _get_invocation_params(self, stop=None, **kwargs):
        params = super()._get_invocation_params(**kwargs)
        if "stop" in params:
            params.pop("stop")
        return params
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

# Enhanced system prompt (unchanged)
SYSTEM_PROMPT = f"""
You are EnerBot, an expert Georgian electricity market data analyst with advanced data visualization intelligence.
...
{DB_SCHEMA_DOC}
"""

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

# === chart detection and processing helpers remain unchanged ===
# (is_chart_request, intelligent_chart_type_selection, process_sql_results_for_chart, extract_json_from_text)
# ---------------------------------------------------------------

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
        schema = db.get_table_info()
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(q: Question, x_app_key: str = Header(...)):
    if not APP_SECRET_KEY or x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # ✅ Use patched GPT-5-mini
        llm = PatchedChatOpenAI(
            model="gpt-5-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )

        db_chain = SQLDatabaseChain.from_llm(
            llm=llm,
            db=db,
            verbose=True,
            use_query_checker=True,
            top_k=50,
            return_intermediate_steps=True
        )

        # Detect chart requests
        is_chart, chart_type = is_chart_request(q.query)
        print(f"Chart detection: is_chart={is_chart}, chart_type={chart_type}, query='{q.query}'")

        result = db_chain.invoke(SYSTEM_PROMPT + "\n\n" + q.query)

        raw_answer = result.get("result", "")
        intermediate_steps = result.get("intermediate_steps", [])

        print(f"Raw answer: {raw_answer}")
        print(f"Intermediate steps: {intermediate_steps}")

        chart_data = []
        chart_metadata = {}
        final_answer = raw_answer

        if is_chart:
            json_data = extract_json_from_text(str(raw_answer))
            if json_data:
                chart_data = json_data
                chart_metadata = {
                    "title": "Energy Data Visualization",
                    "xAxisTitle": "Category",
                    "yAxisTitle": "Value",
                    "datasetLabel": "Data"
                }
                final_answer = "Here's your data visualization:"
            else:
                for step in intermediate_steps:
                    if isinstance(step, dict) and 'sql_cmd' in step:
                        try:
                            sql_query = step['sql_cmd']
                            with engine.connect() as conn:
                                result_proxy = conn.execute(text(sql_query))
                                raw_results = result_proxy.fetchall()
                                print(f"Raw SQL results: {raw_results}")

                                if raw_results:
                                    chart_data, chart_metadata = process_sql_results_for_chart(raw_results, q.query)
                                    chart_type = intelligent_chart_type_selection(raw_results, q.query, chart_type)
                                    final_answer = "Here's your data visualization:"
                                    break
                        except Exception as e:
                            print(f"Error executing SQL directly: {e}")
                            continue

        response = {
            "answer": final_answer if not chart_data else "Here's your data visualization:",
            "chartType": chart_type if chart_data else None,
            "data": chart_data if chart_data else None,
            "chartMetadata": chart_metadata if chart_data else None
        }

        print(f"Final response: {response}")
        return response

    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
