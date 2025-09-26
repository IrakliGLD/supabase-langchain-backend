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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enerbot")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

if not all([OPENAI_API_KEY, SUPABASE_DB_URL, APP_SECRET_KEY]):
    raise RuntimeError("One or more essential environment variables are missing.")

# --- Import DB Schema & Joins ---
from context import DB_SCHEMA_DOC, DB_JOINS, scrub_schema_mentions

ALLOWED_TABLES = [
    "energy_balance_long", "entities", "monthly_cpi",
    "price", "tariff_gen", "tech_quantity", "trade", "dates"
]

engine = create_engine(SUPABASE_DB_URL, poolclass=QueuePool, pool_size=10, pool_pre_ping=True)
db = SQLDatabase(engine, include_tables=ALLOWED_TABLES)

# --- FastAPI Application ---
app = FastAPI(title="EnerBot Backend", version="16.9-forecast-fix")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- System Prompts ---
SQL_GENERATOR_PROMPT = f"""
### ROLE ###
You are an expert SQL writer. Your sole purpose is to generate a single, syntactically correct SQL query 
to answer the user's question based on the provided database schema and join information.

### MANDATORY RULES ###
1.  **GENERATE ONLY SQL.** Your final output must be only the SQL query.
2.  Use the `DB_JOINS` dictionary to determine how to join tables.
3.  For any time-series analysis, query for the entire date range requested, or the entire dataset if no range is specified.

### INTERNAL SCHEMA & JOIN KNOWLEDGE ###
{DB_SCHEMA_DOC}
DB_JOINS = {DB_JOINS}
"""

STRICT_SQL_PROMPT = """
You are an SQL generator. 
Your ONLY job is to return a valid SQL query. 
Do not explain, do not narrate, do not wrap in markdown. 
If you cannot answer, return `SELECT 1;`.
"""

ANALYST_PROMPT = """
You are an expert energy market analyst. Your task is to write a clear, concise narrative based *only* 
on the structured data provided to you.

### MANDATORY RULES ###
1.  **NEVER GUESS.** Use ONLY the numbers and facts provided in the "Computed Stats" section.
2.  **NEVER REVEAL INTERNALS.** Do not mention the database, SQL, or technical jargon.
3.  **ALWAYS BE AN ANALYST.** Your response must be a narrative including trends, peaks, lows, 
    seasonality, and forecasts (if available).
4.  **CONCLUDE SUCCINCTLY.** End with a single, short "Key Insight" line.
"""

# --- Pydantic Models ---
class Question(BaseModel):
    query: str = Field(..., max_length=2000)
    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class APIResponse(BaseModel):
    answer: str
    execution_time: Optional[float] = None

# --- Core Helper & Analytics Functions ---
def clean_and_validate_sql(sql: str) -> str:
    if not sql:
        raise ValueError("Generated SQL query is empty.")
    cleaned_sql = re.sub(r"```(?:sql)?\s*|\s*```", "", sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r"--.*?$", "", cleaned_sql, flags=re.MULTILINE)
    cleaned_sql = re.sub(r"\bLIMIT\s+\d+\b", "", cleaned_sql, flags=re.IGNORECASE)
    cleaned_sql = cleaned_sql.strip().removesuffix(";")
    if not cleaned_sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed.")
    return cleaned_sql

def extract_sql_from_steps(steps: List[Any]) -> Optional[str]:
    sql_query = None
    for step in steps:
        action = step[0] if isinstance(step, tuple) else step
        if isinstance(action, AgentAction):
            if action.tool in ["sql_db_query", "sql_db_query_checker"]:
                tool_input = action.tool_input
                if isinstance(tool_input, dict) and 'query' in tool_input:
                    sql_query = tool_input['query']
                elif isinstance(tool_input, str):
                    sql_query = tool_input
    # Regex fallback
    if not sql_query:
        for step in steps:
            text_blob = str(step)
            match = re.search(r"SELECT\s.+?;", text_blob, flags=re.IGNORECASE | re.DOTALL)
            if match:
                sql_query = match.group(0)
                break
    return sql_query

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
        if df[col].dtype == 'object':
            df[col] = df[col].apply(convert_decimal_to_float)
    return df

# Forecasting helpers (simplified OLS)
def forecast_linear_ols(df: pd.DataFrame, date_col: str, value_col: str, target_date: datetime):
    try:
        df = df.copy()
        df = df.dropna(subset=[date_col, value_col])
        if df.empty: return None
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df["t"] = range(len(df))
        X = sm.add_constant(df["t"])
        y = df[value_col].astype(float)
        model = sm.OLS(y, X).fit()
        target_t = len(df) + ((target_date.year - df[date_col].dt.year.max()) * 12)
        X_future = sm.add_constant(pd.Series([target_t]))
        pred = model.get_prediction(X_future).summary_frame(alpha=0.05)
        return {
            "point": float(pred["mean"].iloc[0]),
            "ci_lower": float(pred["mean_ci_lower"].iloc[0]),
            "ci_upper": float(pred["mean_ci_upper"].iloc[0]),
            "r2": model.rsquared,
            "slope_per_month": model.params["t"],
            "n_obs": len(df),
            "target_month": target_date.strftime("%b-%Y"),
        }
    except Exception as e:
        logger.warning(f"OLS failed: {e}")
        return None

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

        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="openai-tools",
            system_message=SQL_GENERATOR_PROMPT,
            top_k=10000,
            handle_parsing_errors=True  # NEW: auto-retry if parsing fails
        )

        logger.info("Step 1: Invoking agent to generate SQL.")
        result = agent.invoke({"input": q.query}, return_intermediate_steps=True)
        sql_query = extract_sql_from_steps(result.get("intermediate_steps", []))

        if not sql_query and "output" in result and "SELECT" in str(result["output"]).upper():
            sql_query = result["output"]

        if not sql_query:
            logger.warning("Agent failed to generate SQL. Retrying with stricter prompt...")
            strict_agent = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_type="openai-tools",
                system_message=STRICT_SQL_PROMPT,
                top_k=10000,
                handle_parsing_errors=True  # NEW
            )
            result_retry = strict_agent.invoke({"input": q.query}, return_intermediate_steps=True)
            sql_query = extract_sql_from_steps(result_retry.get("intermediate_steps", []))

        if not sql_query:
            logger.error("Both attempts failed to generate SQL. Falling back to text output.")
            final_answer = result.get("output", "I was unable to formulate a query to answer the question.")
        else:
            safe_sql = clean_and_validate_sql(sql_query)
            logger.info(f"Step 2: Executing SQL: {safe_sql}")

            with engine.connect() as conn:
                cursor_result = conn.execute(text(safe_sql))
                rows = cursor_result.fetchall()
                columns = list(cursor_result.keys())

            if not rows:
                final_answer = result.get("output", "The query executed successfully but returned no data.")
            else:
                df = coerce_dataframe(rows, columns)

                # Forecast integration
                forecast_note = ""
                if "date" in df.columns:
                    candidate_cols = [c for c in df.columns if c.lower() in ("p_bal_gel", "p_bal", "balancing_price", "price")]
                    value_col = "p_bal_gel" if "p_bal_gel" in df.columns else (candidate_cols[0] if candidate_cols else None)
                    if value_col:
                        fc = forecast_linear_ols(df, date_col="date", value_col=value_col, target_date=datetime(2030, 12, 1))
                        if fc:
                            forecast_note = (
                                f"Forecast for {fc['target_month']}: {fc['point']:.2f} GEL/MWh "
                                f"(95% CI: {fc['ci_lower']:.2f}–{fc['ci_upper']:.2f}), "
                                f"trend slope {fc['slope_per_month']:.4f}/month, R²={fc['r2']:.3f}, n={fc['n_obs']}."
                            )

                # Combine numeric result + LLM narrative
                narrative = llm.predict(ANALYST_PROMPT + f"\n\nComputed Stats:\n{df.describe(include='all').to_dict()}")
                final_answer = (forecast_note + "\n" + narrative).strip()

        return APIResponse(answer=scrub_schema_mentions(final_answer), execution_time=round(time.time() - start_time, 2))
    except Exception as e:
        logger.error(f"FATAL error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")
