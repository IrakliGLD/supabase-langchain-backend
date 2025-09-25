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
# Assumes context.py exists and contains your full DB_SCHEMA_DOC string
try:
    from context import DB_SCHEMA_DOC
except ImportError:
    # Fallback if context.py is not found
    DB_SCHEMA_DOC = "Schema documentation is not available."

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enerbot")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

if not all([OPENAI_API_KEY, SUPABASE_DB_URL, APP_SECRET_KEY]):
    raise RuntimeError("One or more essential environment variables are missing.")

# --- Database ---
engine = create_engine(SUPABASE_DB_URL, poolclass=QueuePool, pool_size=10, pool_pre_ping=True)
db = SQLDatabase(engine)

# --- FastAPI Application ---
app = FastAPI(title="EnerBot Backend", version="18.0-final-complete")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- System Prompts for a Two-Chain Process ---
SQL_GENERATOR_PROMPT_TEMPLATE = f"""
You are an expert PostgreSQL writer. Your sole purpose is to generate a single, syntactically correct SQL query to answer the user's question.
Your response must be ONLY the SQL query, with no additional text, explanation, or markdown.

User Question:
{{question}}

Database Schema:
{{schema}}

SQL Query:
"""

ANALYST_PROMPT_TEMPLATE = """
You are an expert energy market analyst. Your task is to write a clear, concise narrative based *only* on the structured data provided to you.

### MANDATORY RULES ###
1.  **NEVER GUESS.** Use ONLY the numbers and facts provided in the "Computed Stats" section.
2.  **NEVER REVEAL INTERNALS.** Do not mention the database or SQL.
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

class APIResponse(BaseModel):
    answer: str
    execution_time: Optional[float] = None

# --- Core Helper & Analytics Functions ---

def clean_and_validate_sql(sql: str) -> str:
    if not sql: raise ValueError("Generated SQL query is empty.")
    cleaned_sql = re.sub(r"```(?:sql)?\s*|\s*```", "", sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r"--.*?$", "", cleaned_sql, flags=re.MULTILINE)
    cleaned_sql = re.sub(r"\bLIMIT\s+\d+\b", "", cleaned_sql, flags=re.IGNORECASE)
    cleaned_sql = cleaned_sql.strip().removesuffix(";")
    if not cleaned_sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed.")
    return cleaned_sql

def convert_decimal_to_float(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, list): return [convert_decimal_to_float(x) for x in obj]
    return obj

def coerce_dataframe(rows: List[tuple], columns: List[str]) -> pd.DataFrame:
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=columns)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(convert_decimal_to_float)
    return df

def extract_series(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    if df.empty: return out
    date_col = next((c for c in df.columns if pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df[c], errors='coerce'))), None)
    if not date_col: return out
    
    df[date_col] = pd.to_datetime(df[date_col])
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in numeric_cols and c != date_col]
    
    if numeric_cols and cat_cols:
        for val, group in df.groupby(cat_cols[0]):
            out[str(val)] = group[[date_col, numeric_cols[0]]].rename(columns={date_col: "date", numeric_cols[0]: "value"}).sort_values("date").reset_index(drop=True)
    elif numeric_cols:
        out["series"] = df[[date_col, numeric_cols[0]]].rename(columns={date_col: "date", numeric_cols[0]: "value"}).sort_values("date").reset_index(drop=True)
    return out

def analyze_trend(ts: pd.DataFrame) -> Optional[Tuple[str, float]]:
    if ts.empty or len(ts) < 2: return None
    first, last = ts["value"].iloc[0], ts["value"].iloc[-1]
    direction = "increasing" if last > first else "decreasing" if last < first else "stable"
    pct = ((last - first) / abs(first) * 100) if first != 0 else 0.0
    return direction, round(pct, 1)

def find_extremes(ts: pd.DataFrame) -> Tuple[Optional[Dict], Optional[Dict]]:
    if ts.empty or "date" not in ts.columns: return None, None
    max_row, min_row = ts.loc[ts["value"].idxmax()], ts.loc[ts["value"].idxmin()]
    return (
        {"date": max_row["date"].strftime('%Y-%m'), "value": round(max_row["value"], 1)},
        {"date": min_row["date"].strftime('%Y-%m'), "value": round(min_row["value"], 1)}
    )

def compute_seasonality(ts: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(ts) < 24 or "date" not in ts.columns: return None
    try:
        ts_monthly = ts.set_index('date')['value'].asfreq('MS').ffill()
        decomp = sm.tsa.seasonal_decompose(ts_monthly.dropna(), model="additive", period=12)
        seasonal_profile = decomp.seasonal.groupby(decomp.seasonal.index.month).mean()
        peak_month = datetime(2000, seasonal_profile.idxmax(), 1).strftime('%B')
        trough_month = datetime(2000, seasonal_profile.idxmin(), 1).strftime('%B')
        return {"peak_season": peak_month, "trough_season": trough_month}
    except Exception as e:
        logger.warning(f"Seasonality decomposition failed: {e}")
        return None

def infer_unit_from_query(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["price", "tariff", "cost"]): return "GEL"
    if any(w in q for w in ["generation", "consumption", "energy", "trade"]): return "TJ"
    return "units"

def run_full_analysis_pipeline(df: pd.DataFrame, user_query: str) -> Tuple[Dict, str]:
    logger.info("Running full Python analysis pipeline...")
    series_dict = extract_series(df)
    computed = {}
    for name, s in series_dict.items():
        if "date" in s.columns and not s.empty:
            if trend := analyze_trend(s): computed[f"{name} Trend"] = f"{trend[0]} ({trend[1]:+g}%)"
            peak, low = find_extremes(s)
            if peak: computed[f"{name} Peak"] = peak
            if low: computed[f"{name} Low"] = low
            if seasonality := compute_seasonality(s): computed[f"{name} Seasonality"] = seasonality
    unit = infer_unit_from_query(user_query)
    return computed, unit

# --- API Endpoint ---
@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=APIResponse)
def ask(q: Question, x_app_key: str = Header(...)):
    start_time = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Step 1: Generate SQL in a single, direct LLM call
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
        prompt = ChatPromptTemplate.from_template(SQL_GENERATOR_PROMPT_TEMPLATE)
        
        sql_generation_chain = (
            {"schema": lambda x: DB_SCHEMA_DOC, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("Step 1: Generating SQL query...")
        sql_query = sql_generation_chain.invoke(q.query)

        if not sql_query:
            raise ValueError("Chain failed to generate a SQL query.")

        # Step 2: Clean, validate, and execute the query
        safe_sql = clean_and_validate_sql(sql_query)
        logger.info(f"Step 2: Executing cleaned SQL: {safe_sql}")

        with engine.connect() as conn:
            cursor_result = conn.execute(text(safe_sql))
            rows, columns = cursor_result.fetchall(), list(cursor_result.keys())

        if not rows:
            return APIResponse(answer="I was unable to find data for your specific request.", execution_time=round(time.time() - start_time, 2))

        # Step 3: Run the Python analysis pipeline
        df = coerce_dataframe(rows, columns)
        computed, unit = run_full_analysis_pipeline(df, q.query)
        
        # Step 4: Generate the final narrative in a second, direct LLM call
        logger.info("Step 3: Generating final narrative...")
        analyst_prompt = ChatPromptTemplate.from_template(ANALYST_PROMPT_TEMPLATE)
        stats_str = "\n- ".join([f"{k}: {v}" for k, v in computed.items()]) if computed else "No specific trends or stats were calculated."

        narrative_chain = analyst_prompt | llm | StrOutputParser()
        final_answer = narrative_chain.invoke({
            "question": q.query,
            "unit": unit,
            "stats": stats_str
        })

        return APIResponse(answer=final_answer, execution_time=round(time.time() - start_time, 2))

    except Exception as e:
        logger.error(f"FATAL error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")
