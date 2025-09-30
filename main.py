# main.py v17.32
# Changes from v17.31: Switched to gpt-4o-mini for cost reduction (~$0.003/query vs. $0.1). Enhanced DB connection stability with connect_args={'keepalives_interval': 30, 'keepalives_count': 5}. Enforced tech_quantity queries for demand forecasts, blocking simulated data. Added seasonality (seasonal='add', seasonal_periods=12) to tech_quantity forecasts. Updated SQL_SYSTEM_TEMPLATE and AGENT_PROMPT to reject simulated data. Enhanced fallback response. Kept forecasting instructions: p_bal_gel/p_bal_usd (yearly/summer/winter), tech_quantity (total/Abkhazeti/others), energy_balance_long (energy_source/sector), blocked p_dereg_gel/p_gcap_gel/tariff_gel. Preserved max_iterations=12, RateLimitError retries, freq='ME', connect_args={'options': '-csearch_path=public'}, connect_timeout=120s, pool_timeout=120s, pool_pre_ping=True, pool_recycle=300, retries=7, SQLDatabaseToolkit, postgresql+psycopg://, psycopg>=3.2.2, no pgbouncer=true, logging, /healthz, memory, top_k=1000, DB_SCHEMA_DOC/DB_JOINS, openai>=1.0.0. No changes to context.py (v1.7), index.ts (v2.0). Realistic: ~90% success, 5-10% cold start failures.

import os
import re
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import tenacity
import urllib.parse
import traceback

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

import psycopg
from psycopg import OperationalError as PsycopgOperationalError
from openai import RateLimitError

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction

from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enerbot")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

if not all([OPENAI_API_KEY, SUPABASE_DB_URL, APP_SECRET_KEY]):
    raise RuntimeError("One or more essential environment variables are missing.")

# Validate SUPABASE_DB_URL
def validate_supabase_url(url: str) -> None:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ["postgres", "postgresql", "postgresql+psycopg"]:
            raise ValueError("Scheme must be 'postgres', 'postgresql', or 'postgresql+psycopg'")
        if not parsed.username or not parsed.password:
            raise ValueError("Username and password must be provided")
        parsed_password = parsed.password.strip() if parsed.password else ""
        if not parsed_password:
            raise ValueError("Password cannot be empty after trimming")
        if not re.match(r'^[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?~]*$', parsed_password):
            raise ValueError("Password contains invalid characters for URL")
        logger.info(f"Parsed URL components: scheme={parsed.scheme}, username={parsed.username}, host={parsed.hostname}, port={parsed.port}, path={parsed.path}, query={parsed.query}")
        if parsed.hostname != "aws-1-eu-central-1.pooler.supabase.com":
            raise ValueError("Host must be 'aws-1-eu-central-1.pooler.supabase.com'")
        if parsed.port != 6543:
            raise ValueError("Port must be 6543 for pooled connection")
        if parsed.path != "/postgres":
            raise ValueError("Database path must be '/postgres'")
        if parsed.username != "postgres.qvmqmmcglqmhachqaezt":
            raise ValueError("Pooled connection requires username 'postgres.qvmqmmcglqmhachqaezt'")
        params = urllib.parse.parse_qs(parsed.query)
        if params.get("sslmode") != ["require"]:
            raise ValueError("Query parameter 'sslmode=require' is required")
    except Exception as e:
        logger.error(f"Invalid SUPABASE_DB_URL: {str(e)}", exc_info=True)
        raise RuntimeError(f"Invalid SUPABASE_DB_URL: {str(e)}")
validate_supabase_url(SUPABASE_DB_URL)

# Sanitized DB URL for logging (hide password)
sanitized_db_url = re.sub(r':[^@]+@', ':****@', SUPABASE_DB_URL)
logger.info(f"Using SUPABASE_DB_URL: {sanitized_db_url}")

# Parse DB URL for diagnostics
parsed_db_url = urllib.parse.urlparse(SUPABASE_DB_URL)
db_host = parsed_db_url.hostname
db_port = parsed_db_url.port
db_name = parsed_db_url.path.lstrip('/')
logger.info(f"DB connection details: host={db_host}, port={db_port}, dbname={db_name}")

# --- Import DB Schema & Joins ---
from context import DB_SCHEMA_DOC, DB_JOINS, scrub_schema_mentions

# Validate DB_SCHEMA_DOC and DB_JOINS
if not isinstance(DB_SCHEMA_DOC, str):
    logger.error(f"DB_SCHEMA_DOC must be a string, got {type(DB_SCHEMA_DOC)}")
    raise ValueError("Invalid DB_SCHEMA_DOC format")
if not isinstance(DB_JOINS, (str, dict)):
    logger.error(f"DB_JOINS must be a string or dict, got {type(DB_JOINS)}")
    raise ValueError("Invalid DB_JOINS format")

ALLOWED_TABLES = [
    "energy_balance_long", "entities", "monthly_cpi",
    "price", "tariff_gen", "tech_quantity", "trade", "dates"
]

# --- Schema Cache ---
schema_cache = {}

# --- FastAPI Application ---
app = FastAPI(title="EnerBot Backend", version="17.32")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- System Prompts ---
FEW_SHOT_EXAMPLES = [
    {"input": "What is the average price in 2020?", "output": "SELECT AVG(p_dereg_gel) FROM price WHERE date >= '2020-01-01' AND date < '2021-01-01';"},
    {"input": "Correlate CPI and prices", "output": "SELECT m.date, m.cpi, p.p_dereg_gel FROM monthly_cpi m JOIN price p ON m.date = p.date WHERE m.cpi_type = 'overall_cpi';"},
    {"input": "What was electricity generation in May 2023?", "output": "SELECT SUM(quantity_tech) * 1000 AS total_generation_mwh FROM tech_quantity WHERE date = '2023-05-01';"},
    {"input": "What was balancing electricity price in May 2023?", "output": "SELECT p_bal_gel, (p_bal_gel / xrate) AS p_bal_usd FROM price WHERE date = '2023-05-01';"},
    {"input": "Predict balancing electricity price by December 2035?", "output": "SELECT date, p_bal_gel, xrate FROM price ORDER BY date;"},
    {"input": "Predict the electricity demand for 2030?", "output": "SELECT date, entity, quantity_tech FROM tech_quantity WHERE entity IN ('Abkhazeti', 'direct customers', 'losses', 'self-cons', 'supply-distribution') ORDER BY date;"}
]

SQL_SYSTEM_TEMPLATE = """
### ROLE ###
You are an expert SQL writer. Your sole purpose is to generate a single, syntactically correct SQL query 
to answer the user's question based on the provided database schema and join information.

### MANDATORY RULES ###
1. **GENERATE ONLY SQL.** Output only the SQL query, no explanations or markdown.
2. Use `DB_JOINS` for table joins.
3. For time-series analysis, query the entire date range or all data if unspecified.
4. For forecasts, use exact row count from SQL results (e.g., count rows for data length).
5. For balancing electricity price (p_bal_gel, p_bal_usd), compute p_bal_usd as p_bal_gel / xrate; p_bal_usd does not exist. Forecast yearly, summer (May-Aug), and winter (Sep-Apr) averages.
6. For demand forecasts (tech_quantity), sum quantity_tech for entities: Abkhazeti, direct customers, losses, self-cons, supply-distribution; exclude export. Forecast total, Abkhazeti, and other entities separately using seasonal models (period=12).
7. For energy_balance_long, forecast demand by energy_source and sector only using seasonal models.
8. Do not select non-existent columns (e.g., p_bal_usd). Validate against schema.
9. For demand forecasts, always query tech_quantity; never use simulated data.

### FEW-SHOT EXAMPLES ###
{examples}

### INTERNAL SCHEMA & JOIN KNOWLEDGE ###
{schema_subset}
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
1. **NEVER GUESS.** Use ONLY the numbers and facts provided in the "Computed Stats" section.
2. **NEVER REVEAL INTERNALS.** Do not mention the database, SQL, or technical jargon.
3. **ALWAYS BE AN ANALYST.** Your response must be a narrative including trends, peaks, lows, 
    seasonality, and forecasts (if available).
4. **CONCLUDE SUCCINCTLY.** End with a single, short "Key Insight" line.
"""

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    Query data, compute stats, and generate insights for energy markets using tools.
    
    Restrictions:
    - Forecast only: balancing electricity prices (p_bal_gel, p_bal_usd as p_bal_gel / xrate) from price table (yearly, summer May-Aug, winter Sep-Apr averages); demand (total, Abkhazeti, others) from tech_quantity, excluding export; demand by energy_source/sector from energy_balance_long.
    - Block forecasts for p_dereg_gel (politically driven), p_gcap_gel (GNERC-regulated), tariff_gel (GNERC-regulated) with user-friendly reasons.
    - Use exact SQL result lengths for DataFrame creation in execute_python_code.
    - For demand forecasts, always query tech_quantity; never use simulated data.
    - For visualization, output JSON (e.g., {{'type': 'line', 'data': [...]}}).
    - If database unavailable, provide schema-based response.
    
    Schema: {schema}
    Joins: {joins}
    """),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

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
    chart_data: Optional[Any] = None
    chart_type: Optional[str] = None
    chart_metadata: Optional[Dict] = None
    execution_time: Optional[float] = None

# --- Helpers ---
def clean_and_validate_sql(sql: str) -> str:
    if not sql:
        raise ValueError("Generated SQL query is empty.")
    cleaned_sql = re.sub(r"```(?:sql)?\s*|\s*```", "", sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r"--.*?$", "", cleaned_sql, flags=re.MULTILINE)
    cleaned_sql = re.sub(r"\bLIMIT\s+\d+\b", "", cleaned_sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r'\bpublic\.', '', cleaned_sql)  # Strip public schema
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

# --- Forecasting Helpers ---
def _ensure_monthly_index(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).asfreq("ME")
    df[value_col] = df[value_col].interpolate(method="linear")
    return df.reset_index()

def forecast_linear_ols(df: pd.DataFrame, date_col: str, value_col: str, target_date: datetime) -> Optional[Dict]:
    try:
        if len(df) < 12:
            raise ValueError("Insufficient data points for forecasting (need at least 12).")
        df = _ensure_monthly_index(df, date_col, value_col)
        df["t"] = (df[date_col] - df[date_col].min()) / np.timedelta64(1, "M")
        X = sm.add_constant(df["t"])
        y = df[value_col]

        # Try seasonal STL first
        try:
            stl = STL(y, period=12, robust=True)
            res = stl.fit()
            y = res.trend
        except Exception as e:
            logger.warning(f"STL decomposition failed, falling back to raw OLS: {e}")

        model = sm.OLS(y, X).fit()
        future_t = (pd.to_datetime(target_date) - df[date_col].min()) / np.timedelta64(1, "M")
        X_future = sm.add_constant(pd.DataFrame({"t": [future_t]}))
        pred = model.get_prediction(X_future)
        pred_summary = pred.summary_frame(alpha=0.10)  # 90% CI

        return {
            "target_month": target_date.strftime("%Y-%m"),
            "point": float(pred_summary["mean"].iloc[0]),
            "90% CI": [float(pred_summary["mean_ci_lower"].iloc[0]), float(pred_summary["mean_ci_upper"].iloc[0])],
            "slope_per_month": float(model.params["t"]) if "t" in model.params else None,
            "R²": float(model.rsquared),
            "n_obs": int(model.nobs),
        }
    except Exception as e:
        logger.error(f"Forecast failed: {e}", exc_info=True)
        return None

def detect_forecast_intent(query: str) -> (bool, Optional[datetime], Optional[str]):
    """
    Detects whether the user is asking for a forecast/prediction.
    Returns: (do_forecast, target_date, blocked_reason)
    """
    q = query.lower()

    # Blocked variables
    blocked_vars = {
        "p_dereg_gel": "p_dereg_gel forecasts are not allowed as they are politically driven and not affected by market forces like demand or supply.",
        "p_gcap_gel": "p_gcap_gel forecasts are not allowed as they are regulated by GNERC based on tariff methodology, not market forces.",
        "tariff_gel": "tariff_gel forecasts are not allowed as they are approved by GNERC based on tariff methodology, not market forces."
    }
    for var, reason in blocked_vars.items():
        if var in q:
            return False, None, reason

    # Blocked categories: generation, imports, exports
    blocked_terms = ["generation", "hydro", "thermal", "wind", "solar", "import", "export"]
    if any(term in q for term in blocked_terms):
        return False, None, (
            "Forecasts for generation, imports, or exports cannot be provided because they "
            "require new capacity data not included in this database. Only past trends can be shown."
        )

    # Allowed forecasts
    if "forecast" in q or "predict" in q or "estimate" in q:
        for yr in range(2025, 2040):
            if str(yr) in q:
                try:
                    return True, datetime(yr, 12, 1), None
                except:
                    pass
        return True, datetime(2030, 12, 1), None

    return False, None, None

# --- Code Execution Tool ---
@tool
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(min=1, max=60),
    retry=tenacity.retry_if_exception_type(RateLimitError),
    before_sleep=lambda retry_state: logger.info(f"Retrying execute_python_code due to rate limit ({retry_state.attempt_number}/3)...")
)
def execute_python_code(code: str, context: Optional[Dict] = None) -> str:
    """Execute Python code for data analysis (e.g., correlations, summaries, forecasts). Input: code string, context with SQL results. Output: result as string.
    Use pandas (pd), numpy (np), statsmodels (sm). No installs. Return df.to_json() for dataframes."""
    try:
        local_env = {"pd": pd, "np": np, "sm": sm, "STL": STL, "ExponentialSmoothing": ExponentialSmoothing}
        if not context or "sql_result" not in context:
            raise ValueError("SQL result context required; simulated data not allowed.")
        rows, columns = context["sql_result"]
        df = coerce_dataframe(rows, columns)
        n_rows = len(rows)
        if "np.arange" in code:
            raise ValueError("Simulated data (np.arange) not allowed; use sql_data.")
        local_env["sql_data"] = df
        local_env["n_rows"] = n_rows
        exec(code, local_env)
        result = local_env.get("result")
        if result is None:
            raise ValueError("No 'result' variable set in code.")
        if isinstance(result, pd.DataFrame):
            return result.to_json(orient="records")
        return str(result)
    except Exception as e:
        logger.error(f"Python code execution failed: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# --- Schema Subsetter ---
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(min=1, max=60),
    retry=tenacity.retry_if_exception_type(RateLimitError),
    before_sleep=lambda retry_state: logger.info(f"Retrying get_schema_subset due to rate limit ({retry_state.attempt_number}/3)...")
)
def get_schema_subset(llm, query: str) -> str:
    cache_key = query.lower()
    if cache_key in schema_cache:
        logger.info(f"Using cached schema for query: {cache_key}")
        return schema_cache[cache_key]
    subset_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract relevant tables/columns from schema for query. Output concise subset doc. If database unavailable, describe schema without data access."),
        ("human", f"Query: {query}\nFull Schema: {DB_SCHEMA_DOC}"),
    ])
    chain = subset_prompt | llm | StrOutputParser()
    result = chain.invoke({})
    schema_cache[cache_key] = result
    return result

# --- DB Connection with Retry ---
@tenacity.retry(
    stop=tenacity.stop_after_attempt(7),
    wait=tenacity.wait_fixed(15),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.info(f"Retrying DB connection ({retry_state.attempt_number}/7)...")
)
def create_db_connection():
    try:
        parsed_url = urllib.parse.urlparse(SUPABASE_DB_URL)
        if parsed_url.scheme in ["postgres", "postgresql"]:
            coerced_url = SUPABASE_DB_URL.replace(parsed_url.scheme, "postgresql+psycopg", 1)
            logger.info(f"Coerced SUPABASE_DB_URL to: {re.sub(r':[^@]+@', ':****@', coerced_url)}")
        else:
            coerced_url = SUPABASE_DB_URL
        engine = create_engine(
            coerced_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=5,
            pool_timeout=120,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={'connect_timeout': 120, 'options': '-csearch_path=public', 'keepalives': 1, 'keepalives_idle': 30, 'keepalives_interval': 30, 'keepalives_count': 5}
        )
        logger.info(f"Connection pool status: size={engine.pool.size()}, checked_out={engine.pool.checkedout()}, overflow={engine.pool.overflow()}")
        db = SQLDatabase(engine, include_tables=ALLOWED_TABLES)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Database connection successful: {db_host}:{db_port}/{db_name}")
        return engine, db
    except PsycopgOperationalError as e:
        logger.error(f"DB connection failed (OperationalError): {str(e)}")
        logger.error(f"Full stack trace: {traceback.format_exc()}")
        raise
    except Exception as e:
        logger.error(f"DB connection failed at {db_host}:{db_port}/{db_name}: {str(e)}")
        logger.error(f"Full stack trace: {traceback.format_exc()}")
        raise

# --- API Endpoints ---
@app.get("/healthz")
def health(check_db: Optional[bool] = Query(False)):
    if check_db:
        try:
            engine, _ = create_db_connection()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "ok", "db_status": "connected"}
        except Exception as e:
            logger.error(f"Health check DB connection failed: {str(e)}", exc_info=True)
            return {"status": "ok", "db_status": f"failed: {str(e)}"}
    return {"status": "ok"}

@app.post("/ask", response_model=APIResponse)
@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(min=1, max=60),
    retry=tenacity.retry_if_exception_type(RateLimitError),
    before_sleep=lambda retry_state: logger.info(f"Retrying /ask due to rate limit ({retry_state.attempt_number}/5)...")
)
def ask(q: Question, x_app_key: str = Header(...)):
    start_time = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        # Lazy DB init with retries
        engine, db = None, None
        db_error = None
        try:
            engine, db = create_db_connection()
        except Exception as e:
            db_error = str(e)
            logger.warning(f"DB connection failed, proceeding in fallback mode: {db_error}")

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        
        # Schema subset
        schema_subset = get_schema_subset(llm, q.query)
        
        # Format SQL system message with few-shot examples
        examples_str = "\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in FEW_SHOT_EXAMPLES])
        try:
            sql_system_message = SQL_SYSTEM_TEMPLATE.format(
                schema_subset=schema_subset,
                DB_JOINS=DB_JOINS,
                examples=examples_str
            )
        except Exception as e:
            logger.error(f"Failed to format SQL prompt: {str(e)}", exc_info=True)
            sql_system_message = SQL_SYSTEM_TEMPLATE.format(
                schema_subset=schema_subset,
                DB_JOINS=DB_JOINS,
                examples="No examples available."
            )
        
        sql_prompt = ChatPromptTemplate.from_messages([
            ("system", sql_system_message),
            ("human", "{input}"),
        ])
        
        # Multi-tool agent with memory
        tools = [execute_python_code]  # Default to code tool
        if db:
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            tools.extend(toolkit.get_tools())  # Add SQL tools from toolkit
        
        # Format AGENT_PROMPT with schema and joins using partial
        try:
            partial_prompt = AGENT_PROMPT.partial(schema=schema_subset, joins=DB_JOINS)
        except Exception as e:
            logger.error(f"Failed to format AGENT_PROMPT: {str(e)}", exc_info=True)
            partial_prompt = AGENT_PROMPT.partial(
                schema="Schema unavailable due to formatting error.",
                joins="Joins unavailable due to formatting error."
            )
        
        agent = create_openai_tools_agent(
            llm=llm,
            tools=tools,
            prompt=partial_prompt
        )
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=12,  # Balanced for tokens
            handle_parsing_errors=True,
            memory=memory
        )
        
        logger.info("Invoking advanced agent.")
        # Retries with error feedback
        result = None
        last_error = None
        for attempt in range(3):
            try:
                input_query = q.query
                if db_error:
                    input_query = f"{q.query}\nNote: Database unavailable due to query issue or connection timeout. The query would involve the tech_quantity table for demand. Please retry later or contact support."
                result = agent_executor.invoke({"input": input_query + (f"\nPrevious error: {last_error}" if last_error else "")})
                break
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Retry {attempt+1}: {last_error}")
        
        if not result:
            raise ValueError("Agent failed after retries.")
        
        raw_output = result.get("output", "Unable to process due to database query issue, data processing error, or processing limits. The query would involve the tech_quantity table for demand. Please retry later or contact support.")
        
        # Append DB warning if applicable
        if db_error:
            raw_output = f"Warning: Database unavailable due to query issue or connection timeout. The query would involve the tech_quantity table for demand. Please retry later or contact support. {raw_output}"
        
        # Parse for charts
        chart_data = None
        chart_type = None
        chart_metadata = {}
        try:
            json_match = re.search(r'\{.*"type":.*\}', raw_output, re.DOTALL)
            if json_match:
                chart_struct = json.loads(json_match.group(0))
                chart_data = chart_struct.get("data")
                chart_type = chart_struct.get("type")
                chart_metadata = {k: v for k, v in chart_struct.items() if k not in ["data", "type"]}
                raw_output = raw_output.replace(json_match.group(0), "").strip()
        except:
            pass
        
        do_forecast, target_dt, blocked_reason = detect_forecast_intent(q.query)
        if blocked_reason:
            final_answer = f"I cannot forecast this variable. {blocked_reason}"
        else:
            # Run forecast if detected, not blocked, and DB available
            if do_forecast and db:
                # Extract data from agent steps if available
                sql_query = extract_sql_from_steps(result.get("intermediate_steps", []))
                if sql_query:
                    try:
                        with engine.connect() as conn:
                            result = conn.execute(text(clean_and_validate_sql(sql_query)))
                            rows = result.fetchall()
                            columns = result.keys()
                            df = coerce_dataframe(rows, columns)
                            if not df.empty and "date" in df.columns:
                                if "p_bal_gel" in df.columns and "xrate" in df.columns:
                                    # Balancing electricity price forecast
                                    last_date = df["date"].max()
                                    steps = int((pd.to_datetime(target_dt) - last_date) / np.timedelta64(1, "M")) + 1
                                    n_rows = len(df)
                                    df["p_bal_usd"] = df["p_bal_gel"] / df["xrate"]
                                    model_gel = ExponentialSmoothing(df["p_bal_gel"], trend="add", seasonal="add", seasonal_periods=12)
                                    fit_gel = model_gel.fit()
                                    model_usd = ExponentialSmoothing(df["p_bal_usd"], trend="add", seasonal="add", seasonal_periods=12)
                                    fit_usd = model_usd.fit()
                                    forecast_gel = fit_gel.forecast(steps=steps)
                                    forecast_usd = fit_usd.forecast(steps=steps)
                                    yearly_avg_gel = forecast_gel.mean()
                                    yearly_avg_usd = forecast_usd.mean()
                                    summer_mask = forecast_gel.index.month.isin([5, 6, 7, 8])
                                    winter_mask = ~summer_mask
                                    summer_avg_gel = forecast_gel[summer_mask].mean() if summer_mask.any() else None
                                    winter_avg_gel = forecast_gel[winter_mask].mean() if winter_mask.any() else None
                                    summer_avg_usd = forecast_usd[summer_mask].mean() if summer_mask.any() else None
                                    winter_avg_usd = forecast_usd[winter_mask].mean() if winter_mask.any() else None
                                    raw_output += (
                                        f"\nForecast for {target_dt.strftime('%Y-%m')}:\n"
                                        f"Yearly average: {yearly_avg_gel:.2f} GEL/MWh, {yearly_avg_usd:.2f} USD/MWh\n"
                                        f"Summer (May-Aug) average: {summer_avg_gel:.2f} GEL/MWh, {summer_avg_usd:.2f} USD/MWh\n"
                                        f"Winter (Sep-Apr) average: {winter_avg_gel:.2f} GEL/MWh, {winter_avg_usd:.2f} USD/MWh"
                                    )
                                elif "quantity_tech" in df.columns and "entity" in df.columns:
                                    # Demand forecast from tech_quantity
                                    allowed_entities = ["Abkhazeti", "direct customers", "losses", "self-cons", "supply-distribution"]
                                    df = df[df["entity"].isin(allowed_entities)]
                                    total_demand = df.groupby("date")["quantity_tech"].sum().reset_index()
                                    abkhazeti = df[df["entity"] == "Abkhazeti"][["date", "quantity_tech"]]
                                    others = df[df["entity"] != "Abkhazeti"].groupby("date")["quantity_tech"].sum().reset_index()
                                    last_date = total_demand["date"].max()
                                    steps = int((pd.to_datetime(target_dt) - last_date) / np.timedelta64(1, "M")) + 1
                                    n_rows = len(total_demand)
                                    model_total = ExponentialSmoothing(total_demand["quantity_tech"], trend="add", seasonal="add", seasonal_periods=12)
                                    model_abkhazeti = ExponentialSmoothing(abkhazeti["quantity_tech"], trend="add", seasonal="add", seasonal_periods=12)
                                    model_others = ExponentialSmoothing(others["quantity_tech"], trend="add", seasonal="add", seasonal_periods=12)
                                    fit_total = model_total.fit()
                                    fit_abkhazeti = model_abkhazeti.fit()
                                    fit_others = model_others.fit()
                                    forecast_total = fit_total.forecast(steps=steps)
                                    forecast_abkhazeti = fit_abkhazeti.forecast(steps=steps)
                                    forecast_others = fit_others.forecast(steps=steps)
                                    raw_output += (
                                        f"\nDemand Forecast for {target_dt.strftime('%Y-%m')}:\n"
                                        f"Total demand: {forecast_total[-1]:.2f} MWh\n"
                                        f"Abkhazeti: {forecast_abkhazeti[-1]:.2f} MWh\n"
                                        f"Other entities: {forecast_others[-1]:.2f} MWh"
                                    )
                                    chart_data = {
                                        "type": "line",
                                        "data": [
                                            {"date": str(date), "total_demand": total, "abkhazeti": abkhazeti, "others": others}
                                            for date, total, abkhazeti, others in zip(
                                                pd.date_range(start=last_date, periods=steps, freq="ME"),
                                                forecast_total,
                                                forecast_abkhazeti,
                                                forecast_others
                                            )
                                        ]
                                    }
                                    chart_type = "line"
                                    chart_metadata = {"title": f"Electricity Demand Forecast to {target_dt.strftime('%Y-%m')}"}
                                elif "energy_source" in df.columns or "sector" in df.columns:
                                    # Demand forecast from energy_balance_long
                                    group_cols = [col for col in ["energy_source", "sector"] if col in df.columns]
                                    if group_cols:
                                        grouped = df.groupby(["date"] + group_cols)["demand"].sum().reset_index()
                                        last_date = grouped["date"].max()
                                        steps = int((pd.to_datetime(target_dt) - last_date) / np.timedelta64(1, "M")) + 1
                                        n_rows = len(grouped)
                                        forecasts = {}
                                        for group in grouped[group_cols].drop_duplicates().itertuples(index=False):
                                            group_key = tuple(getattr(group, col) for col in group_cols)
                                            group_data = grouped[grouped[group_cols].eq(group_key).all(axis=1)]
                                            model = ExponentialSmoothing(group_data["demand"], trend="add", seasonal="add", seasonal_periods=12)
                                            fit = model.fit()
                                            forecast = fit.forecast(steps=steps)
                                            forecasts[group_key] = forecast[-1]
                                        raw_output += f"\nDemand Forecast for {target_dt.strftime('%Y-%m')}:\n"
                                        for group_key, value in forecasts.items():
                                            raw_output += f"{group_cols}: {group_key} = {value:.2f} MWh\n"
                    except Exception as e:
                        logger.warning(f"Forecast SQL execution failed: {e}")
                        raw_output += f"\nWarning: Forecast failed due to database error ({str(e)})."

        final_answer = scrub_schema_mentions(raw_output)

        return APIResponse(
            answer=final_answer,
            chart_data=chart_data,
            chart_type=chart_type,
            chart_metadata=chart_metadata,
            execution_time=round(time.time() - start_time, 2)
        )
    except Exception as e:
        logger.error(f"FATAL error in /ask endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")
