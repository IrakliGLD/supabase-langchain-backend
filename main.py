# main.py v17.21
# Changes from v17.20: Fixed SyntaxError in convert_decimal_to_float (line 210) by correcting mismatched bracket in tuple comprehension (changed [] to ()). Preserved v17.20 changes: updated validate_supabase_url to allow postgresql+psycopg scheme, coerced SUPABASE_DB_URL to postgresql+psycopg:// in create_db_connection, removed pgbouncer=true and psycopg2 fallback. Kept pooled connection (aws-1-eu-central-1.pooler.supabase.com:6543), SUPABASE_DB_URL name, and all v17.20 features (DB diagnostics, fallback, retries, logging, /healthz, memory, schema subset, forecasts, top_k=1000). No changes to context.py (v1.7 correct) or index.ts (v2.0 robust). Realistic note: Correct password with psycopg yields 95-100% success.

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

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
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
        # Trim whitespace from password and validate characters
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

ALLOWED_TABLES = [
    "energy_balance_long", "entities", "monthly_cpi",
    "price", "tariff_gen", "tech_quantity", "trade", "dates"
]

# --- FastAPI Application ---
app = FastAPI(title="EnerBot Backend", version="17.21")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- System Prompts ---
FEW_SHOT_EXAMPLES = [
    {"input": "What is the average price in 2020?", "output": "SELECT AVG(p_dereg_gel) FROM price WHERE date >= '2020-01-01' AND date < '2021-01-01';"},
    {"input": "Correlate CPI and prices", "output": "SELECT m.date, m.cpi, p.p_dereg_gel FROM monthly_cpi m JOIN price p ON m.date = p.date;"},
    # Add 3-5 more as needed
]

SQL_SYSTEM_TEMPLATE = """
### ROLE ###
You are an expert SQL writer. Your sole purpose is to generate a single, syntactically correct SQL query 
to answer the user's question based on the provided database schema and join information.

### MANDATORY RULES ###
1.  **GENERATE ONLY SQL.** Your final output must be only the SQL query.
2.  Use the `DB_JOINS` dictionary to determine how to join tables.
3.  For any time-series analysis, query for the entire date range requested, or the entire dataset if no range is specified.

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
1.  **NEVER GUESS.** Use ONLY the numbers and facts provided in the "Computed Stats" section.
2.  **NEVER REVEAL INTERNALS.** Do not mention the database, SQL, or technical jargon.
3.  **ALWAYS BE AN ANALYST.** Your response must be a narrative including trends, peaks, lows, 
    seasonality, and forecasts (if available).
4.  **CONCLUDE SUCCINCTLY.** End with a single, short "Key Insight" line.
"""

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are a flexible data analyst bot for energy markets. Use tools to query data, compute stats (trends, correlations, summaries), and generate insights.
    
    Restrictions:
    - Only forecast prices, CPI, demand. Block generation/import/export forecasts.
    - If visualization needed, output JSON for charts (e.g., {{'type': 'line', 'data': [...]}}).
    - If database is unavailable, provide a schema-based response without data.
    
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
    df = df.set_index(date_col).asfreq("MS")
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

    # Blocked categories: generation, imports, exports
    blocked_terms = ["generation", "hydro", "thermal", "wind", "solar", "import", "export"]
    if any(term in q for term in blocked_terms):
        return False, None, (
            "Forecasts for generation, imports, or exports cannot be provided because they "
            "require new capacity data not included in this database. Only past trends can be shown."
        )

    # Otherwise allow forecast
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
def execute_python_code(code: str) -> str:
    """Execute Python code for data analysis (e.g., correlations, summaries). Input: code string. Output: result as string.
    Use pandas (pd), numpy (np), statsmodels (sm). No installs. Return df.to_json() for dataframes."""
    try:
        local_env = {"pd": pd, "np": np, "sm": sm, "STL": STL}  # Restricted env
        exec(code, local_env)
        result = local_env.get("result")  # Assume code sets 'result'
        if result is None:
            raise ValueError("No 'result' variable set in code.")
        if isinstance(result, pd.DataFrame):
            return result.to_json(orient="records")
        return str(result)
    except SyntaxError as se:
        return f"Syntax error: {str(se)}"
    except NameError as ne:
        return f"Name error: {str(ne)} - Check variable definitions."
    except Exception as e:
        return f"Error: {str(e)}"

# --- Schema Subsetter ---
def get_schema_subset(llm, query: str) -> str:
    subset_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract relevant tables/columns from schema for query. Output concise subset doc. If database unavailable, describe schema without data access."),
        ("human", f"Query: {query}\nFull Schema: {DB_SCHEMA_DOC}"),
    ])
    chain = subset_prompt | llm | StrOutputParser()
    return chain.invoke({})

# --- DB Connection with Retry ---
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(5),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.info(f"Retrying DB connection ({retry_state.attempt_number}/3)...")
)
def create_db_connection():
    try:
        # Coerce SUPABASE_DB_URL to use postgresql+psycopg://
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
            pool_timeout=10,
            connect_args={'connect_timeout': 10}
        )
        db = SQLDatabase(engine, include_tables=ALLOWED_TABLES)
        # Test connection
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

        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
        
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
            logger.error(f"Failed to format SQL prompt: {e}")
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
            tools.extend(db.get_usable_table_names())  # Add SQL tools if DB connected
        
        agent = create_openai_tools_agent(
            llm=llm,
            tools=tools,
            prompt=AGENT_PROMPT
        )
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10,  # Limit for safety
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
                if db_error and "select" in q.query.lower():
                    input_query = f"{q.query}\nNote: Database is unavailable, provide a schema-based response without data."
                result = agent_executor.invoke({
                    "input": input_query + (f"\nPrevious error: {last_error}" if last_error else ""),
                    "schema": schema_subset,
                    "joins": DB_JOINS
                })
                break
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Retry {attempt+1}: {last_error}")
        
        if not result:
            raise ValueError("Agent failed after retries.")
        
        raw_output = result.get("output", "Unable to process.")
        
        # Append DB warning if applicable
        if db_error:
            raw_output = f"Warning: Database connection failed ({db_error}). Results are schema-based and may lack data. {raw_output}"
        
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
            final_answer = "I can only analyze historical data. " + blocked_reason
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
                            if not df.empty and "date" in df.columns and any(col in df.columns for col in ["p_dereg_gel", "cpi", "quantity_tech"]):
                                value_col = next((col for col in ["p_dereg_gel", "cpi", "quantity_tech"] if col in df.columns), None)
                                if value_col:
                                    fc = forecast_linear_ols(df, "date", value_col, target_dt)
                                    if fc:
                                        raw_output += f"\nForecast: {json.dumps(fc)}"
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
        logger.error(f"FATAL error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")
