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

from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd
import statsmodels.api as sm

# --- Configuration & Setup ---

# Assumes context.py exists and contains: DB_SCHEMA_DOC = """...your schema..."""
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

# --- Database & Join Knowledge Graph ---
ALLOWED_TABLES = ["energy_balance_long", "entities", "monthly_cpi", "price", "tariff_gen", "tech_quantity", "trade"]
DB_JOINS = {
    "energy_balance_long": {"join_on": "date", "related_to": ["price", "trade"]},
    "price": {"join_on": "date", "related_to": ["energy_balance_long"]},
    "trade": {"join_on": "date", "related_to": ["energy_balance_long"]},
}

engine = create_engine(SUPABASE_DB_URL, poolclass=QueuePool, pool_size=10, pool_pre_ping=True)
db = SQLDatabase(engine, include_tables=ALLOWED_TABLES)

# --- FastAPI Application ---
app = FastAPI(title="EnerBot Backend", version="10.0-final-complete")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- System Prompts ---
SQL_GENERATOR_PROMPT = f"""
### ROLE ###
You are an expert SQL writer. Your sole purpose is to generate a single, syntactically correct SQL query to answer the user's question based on the provided database schema and join information.

### MANDATORY RULES ###
1.  **GENERATE ONLY SQL.** Your final output must be only the SQL query.
2.  Use the `DB_JOINS` dictionary to determine how to join tables.
3.  For any time-series analysis, your query must cover the entire date range requested.

### INTERNAL SCHEMA & JOIN KNOWLEDGE ###
{DB_SCHEMA_DOC}
DB_JOINS = {DB_JOINS}
"""

ANALYST_PROMPT = """
You are an expert energy market analyst. Your task is to write a clear, concise narrative based *only* on the structured data provided to you.

### MANDATORY RULES ###
1.  **NEVER GUESS.** Use ONLY the numbers and facts provided in the "Computed Stats" section.
2.  **NEVER REVEAL INTERNALS.** Do not mention the database, SQL, or technical jargon.
3.  **ALWAYS BE AN ANALYST.** Your response must be a narrative including trends, peaks, lows, and seasonality.
4.  **CONCLUDE SUCCINCTLY.** End with a single, short "Key Insight" line.
"""

# --- Pydantic Models for API ---
class Question(BaseModel):
    query: str = Field(..., max_length=2000)
    @validator("query")
    def validate_query(cls, v):
        if not v.strip(): raise ValueError("Query cannot be empty")
        return v.strip()

class APIResponse(BaseModel):
    answer: str
    execution_time: Optional[float] = None

# --- Core Helper & Analytics Functions (Full Implementation) ---

def clean_and_validate_sql(sql: str) -> str:
    if not sql: raise ValueError("Generated SQL query is empty.")
    cleaned_sql = re.sub(r"```(?:sql)?\s*|\s*```", "", sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r"--.*?$", "", cleaned_sql, flags=re.MULTILINE)
    cleaned_sql = re.sub(r"\bLIMIT\s+\d+\b", "", cleaned_sql, flags=re.IGNORECASE)
    cleaned_sql = cleaned_sql.strip().removesuffix(";")
    if not cleaned_sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed.")
    return cleaned_sql

def extract_sql_from_steps(steps: List[Any]) -> Optional[str]:
    for step in reversed(steps):
        try:
            action = step[0] if isinstance(step, tuple) else step
            if isinstance(action, AgentAction):
                tool_input = action.tool_input
                if isinstance(tool_input, dict) and 'query' in tool_input:
                    return tool_input['query']
                if isinstance(tool_input, str) and "SELECT" in tool_input.upper():
                    return tool_input
        except Exception:
            continue
    return None

def convert_decimal_to_float(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, list): return [convert_decimal_to_float(x) for x in obj]
    if isinstance(obj, tuple): return tuple(convert_decimal_to_float(x) for x in obj)
    if isinstance(obj, dict): return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    return obj

def coerce_dataframe(rows: List[tuple], columns: List[str]) -> pd.DataFrame:
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=columns)
    for col in df.columns:
        # Attempt to convert object columns that might contain Decimals
        if df[col].dtype == 'object':
            df[col] = df[col].apply(convert_decimal_to_float)
    return df

def extract_series(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if df.empty: return out
    
    date_col_name = None
    for c in df.columns:
        try:
            # Check if a column can be converted to datetime
            pd.to_datetime(df[c], errors='raise')
            date_col_name = c
            break
        except (ValueError, TypeError):
            continue

    if date_col_name:
        df[date_col_name] = pd.to_datetime(df[date_col_name])
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if c not in numeric_cols and c != date_col_name]
        
        if numeric_cols and cat_cols:
            value_col = numeric_cols[0]
            category_col = cat_cols[0]
            for cat_val, group in df.groupby(category_col):
                out[str(cat_val)] = group[[date_col_name, value_col]].rename(columns={date_col_name: "date", value_col: "value"}).sort_values("date").reset_index(drop=True)
        elif numeric_cols:
            value_col = numeric_cols[0]
            out["series"] = df[[date_col_name, value_col]].rename(columns={date_col_name: "date", value_col: "value"}).sort_values("date").reset_index(drop=True)
    elif len(df.columns) >= 2:
        # Fallback for non-timeseries data (e.g., summary GROUP BY)
        label_col, value_col = df.columns[0], df.columns[1]
        out["categories"] = df[[label_col, value_col]].rename(columns={label_col: "label", value_col: "value"}).reset_index(drop=True)
    return out

def analyze_trend(ts: pd.DataFrame) -> Optional[Tuple[str, float]]:
    if ts.empty or len(ts) < 2: return None
    first, last = ts["value"].iloc[0], ts["value"].iloc[-1]
    direction = "increasing" if last > first else "decreasing" if last < first else "stable"
    pct = ((last - first) / abs(first) * 100) if first != 0 else 0.0
    return direction, round(pct, 1)

def find_extremes(ts: pd.DataFrame) -> Tuple[Optional[Dict], Optional[Dict]]:
    if ts.empty or "date" not in ts.columns: return None, None
    max_row = ts.loc[ts["value"].idxmax()]
    min_row = ts.loc[ts["value"].idxmin()]
    return (
        {"date": max_row["date"].strftime('%Y-%m'), "value": round(max_row["value"], 1)},
        {"date": min_row["date"].strftime('%Y-%m'), "value": round(min_row["value"], 1)}
    )

def compute_seasonality(ts: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(ts) < 24 or "date" not in ts.columns: return None
    try:
        ts_monthly = ts.set_index('date')['value'].asfreq('MS', method='ffill')
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
    if any(w in q for w in ["capacity", "power"]): return "MW"
    return "units"

def run_full_analysis_pipeline(df: pd.DataFrame, user_query: str) -> Tuple[Dict, Dict, str]:
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

def generate_final_narrative(llm: ChatOpenAI, user_query: str, unit: str, computed: Dict) -> str:
    stats_str = "\n- ".join([f"{k}: {v}" for k, v in computed.items()])
    prompt = f"User query: {user_query}\nUnit: {unit}\nComputed Stats:\n- {stats_str}"
    msg = llm.invoke([{"role": "system", "content": ANALYST_PROMPT}, {"role": "user", "content": prompt}])
    return getattr(msg, "content", "Could not generate a narrative.").strip()

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
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True, agent_type="openai-tools", system_message=SQL_GENERATOR_PROMPT)

        logger.info("Invoking agent to generate SQL...")
        result = agent.invoke({"input": q.query}, return_intermediate_steps=True)
        
        sql_query = extract_sql_from_steps(result.get("intermediate_steps", []))

        if not sql_query:
            logger.warning("Agent failed to generate SQL. Falling back to its text response.")
            final_answer = result.get("output", "I could not generate a query for that question.")
        else:
            logger.info(f"SQL extracted: {sql_query}")
            safe_sql = clean_and_validate_sql(sql_query)
            
            with engine.connect() as conn:
                cursor_result = conn.execute(text(safe_sql))
                rows = cursor_result.fetchall()
                columns = list(cursor_result.keys())

            if not rows:
                final_answer = "Based on the available data, I couldn't find any information for your specific request."
            else:
                df = coerce_dataframe(rows, columns)
                computed, unit = run_full_analysis_pipeline(df, q.query)
                final_answer = generate_final_narrative(llm, q.query, unit, computed)
        
        return APIResponse(answer=final_answer, execution_time=round(time.time() - start_time, 2))

    except Exception as e:
        logger.error(f"FATAL error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")
