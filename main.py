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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd
import statsmodels.api as sm

# --- Configuration & Setup ---
try:
    from context import DB_SCHEMA_DOC
except ImportError:
    DB_SCHEMA_DOC = "Schema documentation is not available."

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enerbot")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

if not all([OPENAI_API_KEY, SUPABASE_DB_URL, APP_SECRET_KEY]):
    raise RuntimeError("One or more essential environment variables are missing.")

# --- Database & Join Knowledge ---
ALLOWED_TABLES = ["energy_balance_long", "entities", "monthly_cpi", "price", "tariff_gen", "tech_quantity", "trade"]
engine = create_engine(SUPABASE_DB_URL, poolclass=QueuePool, pool_size=10, pool_pre_ping=True)
db = SQLDatabase(engine, include_tables=ALLOWED_TABLES)

# --- FastAPI Application ---
app = FastAPI(title="EnerBot Backend", version="15.0-lcel-architect")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- System Prompts ---
SQL_GENERATOR_PROMPT_TEMPLATE = """
You are an expert PostgreSQL writer. Your sole purpose is to generate a single, syntactically correct SQL query to answer the user's question.
Use the provided database schema to inform your query. Do not add any explanation or narrative.

User Question:
{question}

Database Schema:
{schema}

SQL Query:
"""

ANALYST_PROMPT_TEMPLATE = """
You are an expert energy market analyst. Your task is to write a clear, concise narrative based *only* on the structured data provided to you.

### MANDATORY RULES ###
1.  **NEVER GUESS.** Use ONLY the numbers and facts provided in the "Computed Stats" section.
2.  **NEVER REVEAL INTERNALS.** Do not mention the database, SQL, or technical jargon.
3.  **ALWAYS BE AN ANALYST.** Your response must be a narrative including trends, peaks, lows, and seasonality.
4.  **CONCLUDE SUCCINCTLY.** End with a single, short "Key Insight" line.

User's Original Question: {question}
Unit for Analysis: {unit}
Computed Stats:
{stats}
"""

# --- Pydantic Models ---
class Question(BaseModel):
    query: str = Field(..., max_length=2000)

# --- Core Helper & Analytics Functions ---
# (Full implementation of all helper functions: clean_and_validate_sql, coerce_dataframe, extract_series, analyze_trend, etc.)
def clean_and_validate_sql(sql: str) -> str:
    if not sql: raise ValueError("Generated SQL query is empty.")
    cleaned_sql = re.sub(r"```(?:sql)?\s*|\s*```", "", sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r"--.*?$", "", cleaned_sql, flags=re.MULTILINE)
    cleaned_sql = re.sub(r"\bLIMIT\s+\d+\b", "", cleaned_sql, flags=re.IGNORECASE)
    cleaned_sql = cleaned_sql.strip().removesuffix(";")
    if not cleaned_sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed.")
    return cleaned_sql

# ... [Include all other helper and analytics functions here] ...
def run_full_analysis_pipeline(df: pd.DataFrame, user_query: str) -> Tuple[Dict, str]:
    # This function remains the same as the previous correct version
    return {}, "units"

# --- API Endpoint ---
@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(q: Question, x_app_key: str = Header(...)):
    start_time = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # --- New LCEL Architecture ---
        
        # Step 1: Define the SQL Generation Chain
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
        prompt = ChatPromptTemplate.from_template(SQL_GENERATOR_PROMPT_TEMPLATE)
        
        sql_generation_chain = (
            {"schema": db.get_table_info, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Step 2: Generate the SQL query
        logger.info("Step 1: Generating SQL query with LCEL chain.")
        sql_query = sql_generation_chain.invoke(q.query)

        if not sql_query:
            raise ValueError("LCEL chain failed to generate a SQL query.")

        # Step 3: Clean, validate, and execute the query
        logger.info(f"Step 2: SQL generated: {sql_query}")
        safe_sql = clean_and_validate_sql(sql_query)
        logger.info(f"Step 3: Executing cleaned SQL: {safe_sql}")

        with engine.connect() as conn:
            cursor_result = conn.execute(text(safe_sql))
            rows = cursor_result.fetchall()
            columns = list(cursor_result.keys())

        if not rows:
            return {"answer": "The query executed successfully but returned no data for your request.", "execution_time": round(time.time() - start_time, 2)}

        # Step 4: Run the Python analysis pipeline
        logger.info(f"Step 4: Full dataset of {len(rows)} rows retrieved. Running analysis.")
        df = pd.DataFrame(rows, columns=columns) # Assuming coerce_dataframe is part of your pipeline
        # computed, unit = run_full_analysis_pipeline(df, q.query)
        computed, unit = {}, "units" # Placeholder for your analysis function
        
        # Step 5: Generate the final narrative with a separate chain
        analyst_prompt = ChatPromptTemplate.from_template(ANALYST_PROMPT_TEMPLATE)
        stats_str = "\n- ".join([f"{k}: {v}" for k, v in computed.items()]) if computed else "No specific trends or stats were calculated."

        narrative_chain = analyst_prompt | llm | StrOutputParser()
        final_answer = narrative_chain.invoke({
            "question": q.query,
            "unit": unit,
            "stats": stats_str
        })

        return {"answer": final_answer, "execution_time": round(time.time() - start_time, 2)}

    except Exception as e:
        logger.error(f"FATAL error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")
