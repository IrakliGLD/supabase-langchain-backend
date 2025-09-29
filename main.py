# === main.py v18.2-fallback ===
# Changes vs v18.1-diagnostics:
# - Auto-fallback from transaction pooler (6543) -> session pooler (5432) when "connection refused"
# - Centralized engine builder + ensure_engine_ready() used by /db_ping and /ask
# - Keeps SSL, connect_timeout, diagnostics, and hybrid analysis

import os
import re
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine.url import make_url, URL

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enerbot")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL_RAW = os.getenv("SUPABASE_DB_URL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "false").lower() in ("1", "true", "yes")

if not all([OPENAI_API_KEY, SUPABASE_DB_URL_RAW, APP_SECRET_KEY]):
    raise RuntimeError("One or more essential environment variables are missing.")

# --- Import DB Schema & Joins ---
from context import DB_SCHEMA_DOC, DB_JOINS, STRUCTURED_SCHEMA, COLUMN_LABELS, scrub_schema_mentions

ALLOWED_TABLES = STRUCTURED_SCHEMA["tables"]
ALLOWED_COLUMNS = set(STRUCTURED_SCHEMA["columns"])

# --- DB URL normalization (force SSL, add connect timeout) ---
def _normalize_db_url(u: str) -> str:
    if not u:
        return u
    if "sslmode=" not in u:
        u += ("&" if "?" in u else "?") + "sslmode=require"
    if "connect_timeout=" not in u:
        u += ("&" if "?" in u else "?") + "connect_timeout=8"
    return u

SUPABASE_DB_URL = _normalize_db_url(SUPABASE_DB_URL_RAW)

# --- Engine builder & fallback ---
def _build_engine(url: str):
    return create_engine(
        url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=2,
        pool_pre_ping=True,
        pool_recycle=1800,  # recycle stale conns; pgbouncer-friendly
    )

engine = _build_engine(SUPABASE_DB_URL)

def _is_tx_pooler_url(u: URL) -> bool:
    host = (u.host or "").lower()
    return "pooler.supabase.com" in host and (u.port == 6543)

def _to_session_pooler(u: URL) -> URL:
    # same host but switch to session pooler port=5432
    return u.set(port=5432)

def ensure_engine_ready() -> None:
    """
    Try a quick SELECT 1. If connection is refused on transaction pooler (6543),
    rebuild engine pointing to session pooler (5432) and test again.
    """
    global engine, SUPABASE_DB_URL
    try:
        with engine.connect() as c:
            c.execute(text("select 1"))
            return
    except Exception as e:
        msg = str(e)
        logger.warning("Initial DB ping failed: %s", msg)

        # Only attempt fallback when it's the transaction pooler refusing
        try:
            url = make_url(SUPABASE_DB_URL)
        except Exception:
            url = None

        refused = ("connection refused" in msg.lower()) or ("refused" in msg.lower())
        is_tx_pooler = (url is not None and _is_tx_pooler_url(url))

        if refused and is_tx_pooler:
            logger.warning("Transaction pooler (6543) refused; falling back to Session Pooler (5432).")
            try:
                new_url = _to_session_pooler(url)
                SUPABASE_DB_URL = str(new_url)
                engine.dispose(close=False)
                engine = _build_engine(SUPABASE_DB_URL)
                with engine.connect() as c2:
                    c2.execute(text("select 1"))
                logger.info("Fallback to Session Pooler succeeded.")
                return
            except Exception as e2:
                logger.error("Fallback to Session Pooler failed: %s", e2, exc_info=True)
                raise
        # If not a pooler refusal case, re-raise original
        raise

# --- FastAPI Application ---
app = FastAPI(title="EnerBot Backend", version="18.2-fallback")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- System Prompts ---
SQL_GENERATOR_PROMPT = f"""
### ROLE ###
You generate a single, syntactically correct SQL SELECT query for Postgres based on the user's request,
using ONLY the allowed tables and typical joins below. Return ONLY the SQL.

### RULES ###
- Output ONLY a single SELECT query; no explanations, no markdown.
- Use the joins from DB_JOINS when combining tables.
- Prefer monthly time series if `date` exists; if not, use appropriate keys.
- Use ONLY these tables: {ALLOWED_TABLES}
- Do NOT use tables not in the list above.
- Do NOT use INSERT/UPDATE/DELETE/DDL.
- If unsure, choose the smallest number of tables needed.

### CONTEXT ###
{DB_SCHEMA_DOC}

DB_JOINS = {DB_JOINS}
"""

STRICT_SQL_ENDING = "Return only a valid SQL SELECT query. If impossible, return: SELECT 1;"

ANALYST_PROMPT = """
You are an expert energy market analyst. Write a clear, concise narrative based ONLY on the computed stats provided.

MANDATORY RULES:
1) NEVER GUESS. Use ONLY the numbers and facts in "Computed Stats".
2) NEVER REVEAL INTERNALS. Do not mention databases, tables, SQL, or technical jargon.
3) STRUCTURE YOUR ANSWER:
   - Trend & seasonality (if present)
   - Peaks, troughs, averages, variability
   - Correlations (if any meaningful relationships)
   - Forecast (if provided), including uncertainty
4) CONCLUDE with one brief "Key Insight" line.
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
    data: Optional[Any] = None
    chartType: Optional[str] = None
    chartMetadata: Optional[Dict[str, Any]] = None

# --- Helpers ---
def clean_and_validate_sql(sql: str) -> str:
    if not sql:
        raise ValueError("Generated SQL query is empty.")
    cleaned_sql = re.sub(r"```(?:sql)?\s*|\s*```", "", sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r"--.*?$", "", cleaned_sql, flags=re.MULTILINE)
    cleaned_sql = cleaned_sql.strip().removesuffix(";")
    if not cleaned_sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed.")
    forbidden = [" INSERT ", " UPDATE ", " DELETE ", " DROP ", " ALTER ", " CREATE ", " MERGE "]
    up = f" {cleaned_sql.upper()} "
    if any(tok in up for tok in forbidden):
        raise ValueError("Non-SELECT statements are forbidden.")
    return cleaned_sql

_table_pat = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)

def extract_tables(sql: str) -> List[str]:
    return list({m.group(1) for m in _table_pat.finditer(sql or "")})

def prevalidate_tables(sql: str) -> Tuple[bool, List[str]]:
    tables = extract_tables(sql)
    bad = [t for t in tables if t not in ALLOWED_TABLES]
    return (len(bad) == 0, bad)

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

# --- Analytics Helpers ---
def summary_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if df.empty:
        return {}
    num = df.select_dtypes(include=["number"])
    if num.empty:
        return {}
    desc = num.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    return {k: {m: float(v) for m, v in stat.items()} for k, stat in desc.items()}

def correlation_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    num = df.select_dtypes(include=["number"])
    if num.shape[1] < 2:
        return {}
    corr = num.corr(numeric_only=True).replace({np.nan: None})
    out = {}
    for r in corr.index:
        out[r] = {}
        for c in corr.columns:
            val = corr.loc[r, c]
            out[r][c] = None if pd.isna(val) else float(val)
    return out

def _ensure_monthly_index(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col).asfreq("MS")
    df[value_col] = df[value_col].interpolate(method="linear")
    return df.reset_index()

def forecast_linear_ols(df: pd.DataFrame, date_col: str, value_col: str, target_date: datetime) -> Optional[Dict]:
    try:
        df = _ensure_monthly_index(df, date_col, value_col)
        if df.shape[0] < 12:
            return None

        df["t"] = (df[date_col] - df[date_col].min()) / np.timedelta64(1, "M")
        X = sm.add_constant(df["t"])
        y = df[value_col]

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

# --- Hybrid Forecast Intent Detection ---
FORECAST_KEYWORDS = [
    "forecast", "predict", "projection", "project", "estimate", "expected",
    "future", "next year", "next month", "coming year", "outlook", "where will", "in the future"
]

def _keyword_maybe_forecast(q: str) -> bool:
    ql = q.lower()
    return any(kw in ql for kw in FORECAST_KEYWORDS) or bool(re.search(r"\b20[2-5]\d\b", ql)) or bool(re.search(r"\bin\s+\d+\s+(years?|months?)\b", ql))

def _extract_target_date(q: str, latest_date: Optional[pd.Timestamp]) -> datetime:
    ql = q.lower()
    m = re.search(r"\b(20[2-5]\d)\b", ql)
    if m:
        year = int(m.group(1))
        return datetime(year, 12, 1)
    m = re.search(r"\bin\s+(\d+)\s+(years?|months?)\b", ql)
    if m and latest_date is not None:
        n = int(m.group(1))
        unit = m.group(2)
        base = latest_date.to_pydatetime() if isinstance(latest_date, pd.Timestamp) else latest_date
        if "year" in unit:
            return (base + timedelta(days=365 * n)).replace(day=1)
        return (base + timedelta(days=30 * n)).replace(day=1)
    return datetime(2030, 12, 1)

def llm_confirm_forecast_intent(llm: ChatOpenAI, query: str) -> bool:
    msg = [
        {"role": "system", "content": "You are a classifier. Answer with only 'yes' or 'no'."},
        {"role": "user", "content": f"Does this user request ask for a future forecast or projection? Query: {query}"}
    ]
    try:
        resp = llm.invoke(msg)
        ans = getattr(resp, "content", "").strip().lower()
        return ans.startswith("y")
    except Exception:
        return False

def detect_forecast_intent(query: str, llm_for_check: ChatOpenAI, latest_date: Optional[pd.Timestamp]) -> Tuple[bool, Optional[datetime], Optional[str], Optional[str]]:
    """
    Returns: (do_forecast, target_date, blocked_reason, target_series_hint)
    target_series_hint in {"price","cpi","demand"} or None
    """
    q = query.lower()

    blocked_terms = ["generation", "hydro", "thermal", "wind", "solar", "import", "export"]
    if any(term in q for term in blocked_terms):
        return False, None, (
            "Forecasts for generation, imports, or exports cannot be provided because they "
            "require new capacity data not included in this database. Only past trends can be shown."
        ), None

    if not _keyword_maybe_forecast(query):
        return False, None, None, None

    if not llm_confirm_forecast_intent(llm_for_check, query):
        return False, None, None, None

    if any(w in q for w in ["cpi", "inflation", "price index"]):
        hint = "cpi"
    elif any(w in q for w in ["price", "tariff", "balancing", "gcap", "dereg"]):
        hint = "price"
    elif any(w in q for w in ["demand", "consumption", "quantity"]):
        hint = "demand"
    else:
        hint = None

    target_dt = _extract_target_date(query, latest_date)
    return True, target_dt, None, hint

# --- Chart helpers (lightweight) ---
def make_timeseries_chart_payload(df: pd.DataFrame, date_col: str, value_col: str, title: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    safe = df[[date_col, value_col]].dropna().copy()
    safe[date_col] = pd.to_datetime(safe[date_col]).dt.strftime("%Y-%m-01")
    data = [{"date": d, "value": float(v)} for d, v in safe[[date_col, value_col]].itertuples(index=False, name=None)]
    metadata = {
        "title": title,
        "xAxisTitle": "Month",
        "yAxisTitle": COLUMN_LABELS.get(value_col, "Value"),
        "datasetLabel": "Series",
    }
    return data, metadata, "line"

# --- Health & Diagnostics ---
@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/db_ping")
def db_ping():
    """Quick DB connectivity check without invoking LLMs (with auto-fallback)."""
    try:
        ensure_engine_ready()
        with engine.connect() as c:
            now = c.execute(text("select now()")).scalar()
            ver = c.execute(text("select version()")).scalar()
            return {"ok": True, "now": str(now), "version": str(ver)}
    except Exception as e:
        logger.error("DB ping failed: %s", e, exc_info=True)
        return {"ok": False, "error": str(e)}

@app.get("/diag")
def diag():
    """Sanitized view of DB target (no secrets)."""
    try:
        url = make_url(SUPABASE_DB_URL)
        redacted = url.set(password="***")
        return {
            "ok": True,
            "engine_url": str(redacted),
            "sslmode": "require" if "sslmode=require" in SUPABASE_DB_URL else "unknown",
            "connect_timeout": "8" if "connect_timeout=8" in SUPABASE_DB_URL else "unknown",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# --- Main API Endpoint ---
@app.post("/ask", response_model=APIResponse)
def ask(q: Question, x_app_key: str = Header(...), x_debug: Optional[str] = Header(None)):
    start_time = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        # Ensure DB engine is working (and fallback if needed) before doing any LLM work
        ensure_engine_ready()

        llm_sql = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        llm_analysis = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

        # --- Step 1: Generate SQL (single shot) ---
        logger.info("Generating SQL (single-shot).")
        sql_msg = [
            {"role": "system", "content": SQL_GENERATOR_PROMPT},
            {"role": "user", "content": q.query},
            {"role": "system", "content": STRICT_SQL_ENDING},
        ]
        gen = llm_sql.invoke(sql_msg)
        sql_query = getattr(gen, "content", "").strip()

        # Validate SELECT-only and remove markdown fence
        safe_sql = clean_and_validate_sql(sql_query)

        # Prevalidate tables to avoid obvious mistakes
        ok_tables, bad_tables = prevalidate_tables(safe_sql)
        if not ok_tables:
            hint = f"Use ONLY these tables: {ALLOWED_TABLES}. You used: {bad_tables}. Regenerate."
            gen2 = llm_sql.invoke([
                {"role": "system", "content": SQL_GENERATOR_PROMPT + "\n" + hint},
                {"role": "user", "content": q.query},
                {"role": "system", "content": STRICT_SQL_ENDING},
            ])
            sql_query = getattr(gen2, "content", "").strip()
            safe_sql = clean_and_validate_sql(sql_query)

        # --- Step 2: Execute SQL ---
        logger.info(f"Executing SQL: {safe_sql}")
        with engine.connect() as conn:
            cursor_result = conn.execute(text(safe_sql))
            rows = cursor_result.fetchall()
            columns = list(cursor_result.keys())

        if not rows:
            answer = "I couldn't find relevant data for this request."
            return APIResponse(
                answer=scrub_schema_mentions(answer),
                execution_time=round(time.time() - start_time, 2),
            )

        df = coerce_dataframe(rows, columns)

        # Latest date (for relative forecast targets)
        latest_date = None
        if "date" in df.columns:
            try:
                latest_date = pd.to_datetime(df["date"]).max()
            except Exception:
                latest_date = None

        # --- Step 3: Analytics (summary, corr, forecast) ---
        computed: Dict[str, Any] = {}
        computed["summary_stats"] = summary_statistics(df)
        corr = correlation_matrix(df)
        if corr:
            computed["correlations"] = corr

        # Hybrid forecast intent detection
        do_fc, target_dt, blocked_reason, target_hint = detect_forecast_intent(q.query, llm_sql, latest_date)

        chart_payload = None
        chart_meta = None
        chart_type = None

        if blocked_reason:
            computed["forecast_note"] = blocked_reason
        elif do_fc:
            # Select a reasonable value column based on hint or presence
            value_col = None
            lower_cols = [c.lower() for c in df.columns]

            if target_hint == "price":
                for cand in ["p_bal_gel", "p_dereg_gel", "p_gcap_gel", "p_bal_usd", "p_dereg_usd", "p_gcap_usd"]:
                    if cand in lower_cols:
                        value_col = df.columns[lower_cols.index(cand)]
                        break
            elif target_hint == "cpi":
                for cand in ["cpi"]:
                    if cand in lower_cols:
                        value_col = df.columns[lower_cols.index(cand)]
                        break
            elif target_hint == "demand":
                for cand in ["quantity_tech", "quantity", "volume_tj"]:
                    if cand in lower_cols:
                        value_col = df.columns[lower_cols.index(cand)]
                        break

            if value_col is None:
                for c in df.select_dtypes(include=["number"]).columns:
                    value_col = c
                    break

            if value_col and "date" in df.columns:
                fc = forecast_linear_ols(df, date_col="date", value_col=value_col, target_date=target_dt)
                if fc:
                    computed["forecast"] = {value_col: fc}
                    try:
                        chart_payload, chart_meta, chart_type = make_timeseries_chart_payload(df, "date", value_col, title="Historical & Trend")
                    except Exception:
                        pass

        # --- Step 4: Narrative (compact input) ---
        narrative = llm_analysis.invoke([
            {"role": "system", "content": ANALYST_PROMPT},
            {"role": "user", "content": f"Question: {q.query}\n\nComputed Stats: {computed}"}
        ])
        final_answer = getattr(narrative, "content", "") or "Here is the analysis."
        final_answer = scrub_schema_mentions(final_answer)

        return APIResponse(
            answer=final_answer,
            execution_time=round(time.time() - start_time, 2),
            data=chart_payload,
            chartType=chart_type,
            chartMetadata=chart_meta
        )

    except Exception as e:
        msg = str(e)
        logger.error(f"FATAL error in /ask: {msg}", exc_info=True)

        # Optional debug exposure (be careful; toggle off in production)
        debug_header = (x_debug or "").lower() in ("1", "true", "yes") if x_debug is not None else False
        if DEBUG_ERRORS or debug_header:
            raise HTTPException(status_code=500, detail=msg)

        if any(tok in msg.lower() for tok in ["does not exist", "column", "relation"]):
            raise HTTPException(status_code=400, detail="I had trouble matching the requested data fields. Please rephrase your question or specify the metric and time frame.")

        raise HTTPException(status_code=500, detail="An internal processing error occurred.")
