import os
import re
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_core.agents import AgentAction
from langchain_core.tools import Tool

from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Assumes context.py contains your DB_SCHEMA_DOC string
from context import DB_SCHEMA_DOC

# ---------------- Logging & Env ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enerbot")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

if not OPENAI_API_KEY or not SUPABASE_DB_URL or not APP_SECRET_KEY:
    raise RuntimeError("Missing essential environment variables.")

# ---------------- Database & Join Knowledge ----------------
ALLOWED_TABLES = ["energy_balance_long", "entities", "monthly_cpi", "price", "tariff_gen", "tech_quantity", "trade"]
DB_JOINS = {
    "energy_balance_long": {"join_on": "date", "related_to": ["price", "trade"]},
    "price": {"join_on": "date", "related_to": ["energy_balance_long"]},
    "trade": {"join_on": "date", "related_to": ["energy_balance_long"]},
}

engine = create_engine(SUPABASE_DB_URL, poolclass=QueuePool, pool_size=10, max_overflow=20, pool_pre_ping=True, pool_recycle=3600)
db = SQLDatabase(engine, include_tables=ALLOWED_TABLES)

# ---------------- FastAPI App Initialization (THIS WAS THE MISSING PART) ----------------
app = FastAPI(title="EnerBot Backend", version="7.0-final")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- System Prompt ----------------
SYSTEM_PROMPT = f"""
### ROLE ###
You are EnerBot, an expert Georgian electricity market analyst. Your sole purpose is to answer user questions by querying a database and providing clear, data-driven analysis.

### MANDATORY RULES ###
1.  **NEVER GUESS.** Use ONLY the data returned from the database.
2.  **NEVER REVEAL INTERNALS.** Do not mention SQL, schema, table/column names, or the database in your final answer.
3.  **ALWAYS ANALYZE, NEVER DUMP.** Your final output must be a narrative analysis including trends, peaks, lows, and seasonality. Conclude with a single key insight.
4.  **BE RESILIENT.** If an initial query returns no data, re-examine the schema for alternative column names or synonyms (e.g., HPP -> hydro) and try a different query before concluding no data exists.

### CAPABILITIES & SCHEMA INFO ###
- You must interpret user terminology (e.g., TPP -> thermal).
- You MUST use the provided `DB_JOINS` dictionary to determine how to join tables. Do not guess join keys.
- Key datasets: Generation (`tech_quantity`), Consumption (`energy_balance_long`), Prices (`price`), Trade (`trade`).

### INTERNAL SCHEMA (for your reference only) ###
{DB_SCHEMA_DOC}
DB_JOINS = {DB_JOINS}
"""

# ---------------- Pydantic Models ----------------
class Question(BaseModel):
    query: str = Field(..., max_length=2000)
    user_id: Optional[str] = None
    @validator("query")
    def validate_query(cls, v):
        if not v.strip(): raise ValueError("Query cannot be empty")
        return v.strip()

class APIResponse(BaseModel):
    answer: str
    execution_time: Optional[float] = None

# ---------------- Security, Sanitization & Analytics Helpers ----------------
# All of your helper and analytics functions go here.
# (scrub_schema_mentions, clean_sql, validate_sql_is_safe, analyze_trend, etc.)
# This section is unchanged. For brevity, it is represented by this comment.
# Make sure to copy all your functions into this section.

def scrub_schema_mentions(text: str) -> str:
    # ... (implementation)
    return text

def clean_sql(sql: str) -> str:
    # ... (implementation)
    return sql

def validate_sql_is_safe(sql: str) -> str:
    # ... (implementation)
    return sql

# (And so on for all your other helper functions...)

def process_and_analyze_data(sql: str, user_query: str, analyst_llm: ChatOpenAI) -> str:
    # ... (full implementation from previous step)
    return "Analysis results."

def extract_sql_from_steps(steps: List[Any]) -> Optional[str]:
    # ... (full implementation from previous step)
    return "SELECT ..."

# ---------------- API Endpoints ----------------
@app.get("/healthz")
def health():
    with engine.connect() as conn: conn.execute(text("SELECT 1"))
    return {"ok": True, "ts": datetime.utcnow()}

@app.post("/ask", response_model=APIResponse)
def ask(q: Question, x_app_key: str = Header(...)):
    start_time = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY, request_timeout=60)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        all_tools = toolkit.get_tools()

        original_sql_tool = next((tool for tool in all_tools if tool.name == "sql_db_query"), None)
        
        final_tools = all_tools
        if original_sql_tool:
            def wrapped_sql_run_func(query: str):
                logger.info(f"WRAPPED TOOL: Intercepted query: {query}")
                cleaned_query = clean_sql(query)
                validate_sql_is_safe(cleaned_query)
                return original_sql_tool.run(cleaned_query)

            wrapped_tool = Tool(
                name="sql_db_query",
                func=wrapped_sql_run_func,
                description=original_sql_tool.description
            )
            final_tools = [wrapped_tool if tool.name == "sql_db_query" else tool for tool in all_tools]

        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            tools=final_tools,
            verbose=True,
            agent_type="openai-tools",
            system_message=SYSTEM_PROMPT,
            max_iterations=8,
            early_stopping_method="generate",
        )

        result = agent.invoke({"input": q.query}, return_intermediate_steps=True)
        sql = extract_sql_from_steps(result.get("intermediate_steps", []))

        if not sql:
            logger.warning("No SQL extracted, falling back to agent's direct output.")
            final_answer = scrub_schema_mentions(result.get("output", "I could not determine how to answer that question."))
        else:
            final_answer = process_and_analyze_data(sql, q.query, llm)
        
        return APIResponse(answer=final_answer, execution_time=round(time.time() - start_time, 2))

    except Exception as e:
        logger.error(f"Processing error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")
