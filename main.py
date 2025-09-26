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
from sqlalchemy.pool import QueuePool

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_core.agents import AgentAction
from langchain.callbacks.base import BaseCallbackHandler

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

# --- Import DB Schema & Joins & scrubber from context.py ---
from context import DB_SCHEMA_DOC, DB_JOINS, scrub_schema_mentions

ALLOWED_TABLES = [
    "energy_balance_long", "entities", "monthly_cpi",
    "price", "tariff_gen", "tech_quantity", "trade", "dates"
]

engine = create_engine(SUPABASE_DB_URL, poolclass=QueuePool, pool_size=10, pool_pre_ping=True)
db = SQLDatabase(engine, include_tables=ALLOWED_TABLES)

# --- Callback Handler to Reliably Capture SQL the Agent Executes ---
class SQLCaptureHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_query: Optional[str] = None

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        if action.tool == "sql_db_query":
            if isinstance(action.tool_input, dict):
                self.sql_query = action.tool_input.get("query")
            elif isinstance(action.tool_input, str):
                self.sql_query = action.tool_input

# --- FastAPI Application ---
app = FastAPI(title="EnerBot Backend", version="16.12-predict-now")
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

# --- Helpers ---
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

def extract_sql_from_text_blob(blob: str) -> Optional[str]:
    if not blob:
        return None
    m = re.search(r"SELECT\s+[\s\S]*?;", blob, flags=re.IGNORECASE)
    return m.group(0) if m else None

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

# --- Forecasting ---
def forecast_linear_ols(df: pd.DataFrame, date_col: str, value_col: str, target_date: datetime) -> Optional[Dict[str, Any]]:
    try:
        dfa = df[[date_col, value_col]].dropna().copy()
        if dfa.empty:
            return None
        dfa[date_col] = pd.to_datetime(dfa[date_col])
        dfa = dfa.sort_values(date_col)
        # numeric time index in months since start
        dfa["t"] = (dfa[date_col] - dfa[date_col].min()) / np.timedelta64(1, "M")
        X = sm.add_constant(dfa["t"].values)
        y = dfa[value_col].astype(float).values
        model = sm.OLS(y, X).fit()
        t_future = (pd.to_datetime(target_date) - dfa[date_col].min()) / np.timedelta64(1, "M")
        X_future = sm.add_constant([t_future])
        pred = model.get_prediction(X_future).summary_frame(alpha=0.05)
        return {
            "target_month": target_date.strftime("%Y-%m"),
            "point": float(pred["mean"].iloc[0]),
            "ci_lower": float(pred["mean_ci_lower"].iloc[0]),
            "ci_upper": float(pred["mean_ci_upper"].iloc[0]),
            "slope_per_month": float(model.params[1]),
            "r2": float(model.rsquared),
            "n_obs": int(len(dfa)),
        }
    except Exception as e:
        logger.warning(f"Forecast failed: {e}")
        return None

def generate_narrative(llm: ChatOpenAI, user_query: str, computed: Dict) -> str:
    stats_str = "\n- ".join([f"{k}: {v}" for k, v in computed.items()]) if computed else "No stats computed."
    prompt = f"User query: {user_query}\nComputed Stats:\n- {stats_str}"
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

        # Agent with callback capture + parse error resilience
        sql_capture = SQLCaptureHandler()
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="openai-tools",
            system_message=SQL_GENERATOR_PROMPT,
            top_k=10000,
            handle_parsing_errors=True,
            callbacks=[sql_capture],
        )

        logger.info("Step 1: Invoking agent to generate SQL.")
        result = agent.invoke({"input": q.query}, return_intermediate_steps=True)
        sql_query = sql_capture.sql_query

        # Fallback: try extracting from agent output if callback missed it
        if not sql_query:
            sql_query = extract_sql_from_text_blob(str(result.get("output", "")))

        # Retry with strict prompt if still nothing
        if not sql_query:
            logger.warning("Agent failed to generate SQL. Retrying with STRICT prompt...")
            strict_capture = SQLCaptureHandler()
            strict_agent = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_type="openai-tools",
                system_message=STRICT_SQL_PROMPT,
                top_k=10000,
                handle_parsing_errors=True,
                callbacks=[strict_capture],
            )
            result_retry = strict_agent.invoke({"input": q.query}, return_intermediate_steps=True)
            sql_query = strict_capture.sql_query or extract_sql_from_text_blob(str(result_retry.get("output", "")))

        if not sql_query:
            logger.error("Both attempts failed to generate SQL. Falling back to text output.")
            final_answer = result.get("output", "I was unable to formulate a query to answer the question.")
            return APIResponse(answer=scrub_schema_mentions(final_answer), execution_time=round(time.time() - start_time, 2))

        safe_sql = clean_and_validate_sql(sql_query)
        logger.info(f"Step 2: Executing SQL: {safe_sql}")

        with engine.connect() as conn:
            cursor_result = conn.execute(text(safe_sql))
            rows = cursor_result.fetchall()
            columns = list(cursor_result.keys())

        if not rows:
            final_answer = result.get("output", "The query executed successfully but returned no data.")
            return APIResponse(answer=scrub_schema_mentions(final_answer), execution_time=round(time.time() - start_time, 2))

        # --- Analysis & Forecast (deterministic) ---
        df = coerce_dataframe(rows, columns)

        forecast_line = ""
        try:
            # If we have balancing price, ALWAYS compute a Dec-2030 forecast
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            value_col = next((c for c in df.columns if c.lower() in ("p_bal_gel", "p_bal", "balancing_price", "price")), None)
            if value_col and "date" in df.columns:
                fc = forecast_linear_ols(df, date_col="date", value_col=value_col, target_date=datetime(2030, 12, 1))
                if fc:
                    forecast_line = (
                        f"Forecast for {fc['target_month']}: {fc['point']:.2f} GEL/MWh "
                        f"(95% CI: {fc['ci_lower']:.2f}–{fc['ci_upper']:.2f}); "
                        f"slope {fc['slope_per_month']:.4f}/month, R²={fc['r2']:.3f}, n={fc['n_obs']}."
                    )
        except Exception as e:
            logger.warning(f"Forecast block failed: {e}")

        # Minimal computed stats (optional)
        computed = {}
        if forecast_line:
            computed["Balancing Price Forecast (2030-12)"] = forecast_line

        narrative = generate_narrative(llm, q.query, computed)
        final_answer = (forecast_line + ("\n" if forecast_line else "") + narrative).strip()

        return APIResponse(answer=scrub_schema_mentions(final_answer), execution_time=round(time.time() - start_time, 2))

    except Exception as e:
        logger.error(f"FATAL error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")
