import os
import re
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union

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
from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd

# Internal schema doc (DO NOT expose)
from context import DB_SCHEMA_DOC

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enerbot")

# ---------------- Env ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

if not OPENAI_API_KEY or not SUPABASE_DB_URL or not APP_SECRET_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY, SUPABASE_DB_URL, or APP_SECRET_KEY")

# ---------------- Database ----------------
ALLOWED_TABLES = [
    "energy_balance_long",
    "entities",
    "monthly_cpi",
    "price",
    "tariff_gen",
    "tech_quantity",
    "trade",
]

engine = create_engine(
    SUPABASE_DB_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# LangChain DB wrapper limited to allowed tables
db = SQLDatabase(engine, include_tables=ALLOWED_TABLES)

# ---------------- FastAPI ----------------
app = FastAPI(title="EnerBot Backend", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- System Prompt (for the agent that writes SQL) ----------------
SYSTEM_PROMPT = f"""
You are EnerBot, an autonomous Georgian electricity market analyst.

=== STRICT RULES ===
- Your ONLY tool is to generate a safe SQL SELECT to answer the user's question.
- Do NOT include markdown code fences. Output plain SQL when using the SQL tool.
- NEVER mention or output schema/table/column names to the user (internal only).
- Prefer broad date ranges (e.g., 2015–present) unless the user specifies otherwise.
- When the user asks for "trend", "forecast", "expected" etc., fetch a full time series (no tiny LIMITs).
- Use correct filters and aggregations (SUM/AVG) as needed.

=== INTERNAL SCHEMA (use internally; never reveal) ===
{DB_SCHEMA_DOC}
"""

# ---------------- Models ----------------
class Question(BaseModel):
    query: str = Field(..., max_length=2000)
    user_id: Optional[str] = None

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class APIResponse(BaseModel):
    answer: str
    execution_time: Optional[float] = None


# ---------------- Utils & Scrubbers ----------------
TECH_TERM_MAP = {
    "p_bal_gel": "balancing electricity price (GEL)",
    "p_dereg_gel": "deregulated electricity price (GEL)",
    "p_gcap_gel": "guaranteed capacity price (GEL)",
    "tariff_gel": "tariff (GEL)",
    "volume_tj": "energy volume (TJ)",
    "quantity_tech": "technology quantity",
    "xrate": "exchange rate",
}

def scrub_schema_mentions(text: str) -> str:
    """Remove schema/table/column leaks & technical jargon from final output."""
    if not text:
        return text

    # Replace known column tokens with human labels
    for k, v in TECH_TERM_MAP.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text, flags=re.IGNORECASE)

    # Replace table names with "the database"
    for t in ALLOWED_TABLES:
        text = re.sub(rf"\b{re.escape(t)}\b", "the database", text, flags=re.IGNORECASE)

    # Replace db-technical words carefully
    text = re.sub(r"\b(schema|table|column|sql|join|primary key|foreign key)\b", "data", text, flags=re.IGNORECASE)

    # Clean repetitive 'the database the database' artifacts
    text = re.sub(r"(the database\s+){2,}", "the database ", text)

    return text.strip()


def convert_decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, list):
        return [convert_decimal_to_float(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(convert_decimal_to_float(x) for x in obj)
    if isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    return obj


def infer_unit_from_query(query: str) -> Optional[str]:
    q = query.lower()
    if any(w in q for w in ["price", "tariff", "cost"]):
        return "GEL" if "usd" not in q and "dollar" not in q else "USD"
    if any(w in q for w in ["generation", "consumption", "energy", "trade", "import", "export"]):
        return "TJ"
    if any(w in q for w in ["capacity", "power"]):
        return "MW"
    return None


# ---------------- Memory (read-only: last 3 turns) ----------------
def get_recent_history(user_id: str, limit_pairs: int = 3) -> List[Dict[str, str]]:
    if not user_id:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT role, content
                    FROM chat_history
                    WHERE user_id = :uid
                    ORDER BY created_at DESC
                    LIMIT :lim
                """),
                {"uid": user_id, "lim": limit_pairs * 2},
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception as e:
        logger.info(f"chat_history not available or failed to read: {e}")
        return []


def build_context(user_id: Optional[str], user_query: str) -> str:
    history = get_recent_history(user_id, 3) if user_id else []
    if not history:
        return user_query
    h = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
    return f"{h}\nUser: {user_query}\nAssistant:"


# ---------------- SQL Sanitizer ----------------
DANGEROUS = [
    "DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER",
    "TRUNCATE", "REPLACE", "EXEC", "EXECUTE", "CALL", "MERGE"
]

def clean_sql(sql: str) -> str:
    if not sql:
        return sql
    # strip markdown fences & comments
    sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```\s*", "", sql)
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = sql.strip()
    # ensure single statement
    if sql.endswith(";"):
        sql = sql[:-1]
    return sql.strip()


def validate_sql_is_safe(sql: str) -> None:
    up = sql.upper()
    if not up.startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed.")
    for word in DANGEROUS:
        if re.search(rf"\b{word}\b", up):
            raise ValueError(f"Dangerous SQL operation detected: {word}")
    # optional: basic guard that at least one allowed table is present in FROM/JOIN
    allowed_pat = r"|".join([re.escape(t) for t in ALLOWED_TABLES])
    if not re.search(rf"\b({allowed_pat})\b", up):
        # Allow queries against allowed views if your schema has them; otherwise enforce:
        logger.info("SQL does not reference an allowed table by name; continuing (the agent may select via subquery/view).")


def try_expand_limit_for_timeseries(sql: str, df: pd.DataFrame) -> str:
    """If the agent added a small LIMIT and we got very few rows, try to remove/raise it."""
    if df is None or df.shape[0] >= 24:
        return sql
    if re.search(r"\bLIMIT\s+\d+\b", sql, flags=re.IGNORECASE):
        sql2 = re.sub(r"\bLIMIT\s+\d+\b", "LIMIT 50000", sql, flags=re.IGNORECASE)
        return sql2
    return sql


# ---------------- Data shaping ----------------
def coerce_dataframe(rows: List[tuple]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    # build DataFrame, infer dtypes
    df = pd.DataFrame([list(r) for r in rows])
    # Convert Decimals
    df = df.applymap(convert_decimal_to_float)
    return df


def detect_timeseries(df: pd.DataFrame) -> bool:
    if df.empty or df.shape[1] < 2:
        return False
    # If any column is date-like (datetime/date or parseable) and another numeric → treat as timeseries
    # Heuristics: check if any col can parse to datetime and any other col numeric
    for i in range(df.shape[1]):
        col = df.iloc[:, i]
        # try parse
        try:
            dt = pd.to_datetime(col)
            # numeric exists?
            for j in range(df.shape[1]):
                if j == i:
                    continue
                if pd.api.types.is_numeric_dtype(df.iloc[:, j]):
                    return True
        except Exception:
            continue
    return False


def extract_series(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build one or more series:
    - If two columns: (date/label, value)
    - If three columns and types look like (date, category, value) → multiple series by category
    - If more columns: numeric columns collapsed into separate series if share date.
    Returns dict: {series_name: DataFrame[date,value]} or {label: DataFrame[label,value]} when non-timeseries.
    """
    out: Dict[str, pd.DataFrame] = {}

    # If we can parse any date-like col, prefer it as time axis
    date_col_idx = None
    for i in range(df.shape[1]):
        try:
            pd.to_datetime(df.iloc[:, i])
            date_col_idx = i
            break
        except Exception:
            continue

    if date_col_idx is not None:
        # Time series branch
        date_series = pd.to_datetime(df.iloc[:, date_col_idx], errors="coerce")
        # find numeric cols
        numeric_idx = [j for j in range(df.shape[1]) if j != date_col_idx and pd.api.types.is_numeric_dtype(df.iloc[:, j])]
        # category col (optional)
        cat_idx = None
        for j in range(df.shape[1]):
            if j == date_col_idx:
                continue
            if not pd.api.types.is_numeric_dtype(df.iloc[:, j]):
                cat_idx = j
                break

        if numeric_idx and cat_idx is not None:
            # likely (date, category, value)
            val_col = numeric_idx[0]
            cats = df.iloc[:, cat_idx].astype(str)
            vals = pd.to_numeric(df.iloc[:, val_col], errors="coerce")
            tmp = pd.DataFrame({"date": date_series, "cat": cats, "value": vals}).dropna()
            for label, g in tmp.groupby("cat"):
                s = g[["date", "value"]].sort_values("date").reset_index(drop=True)
                out[str(label)] = s
        elif numeric_idx:
            # single numeric col, one series
            val_col = numeric_idx[0]
            vals = pd.to_numeric(df.iloc[:, val_col], errors="coerce")
            s = pd.DataFrame({"date": date_series, "value": vals}).dropna().sort_values("date").reset_index(drop=True)
            out["series"] = s
        else:
            # no numeric col → return empty
            return {}
    else:
        # Non-time series: expect (label, value)
        if df.shape[1] < 2:
            return {}
        lbl = df.iloc[:, 0].astype(str)
        vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s = pd.DataFrame({"label": lbl, "value": vals}).dropna().reset_index(drop=True)
        out["categories"] = s

    return out


# ---------------- Numeric analytics ----------------
def analyze_trend(ts: pd.DataFrame) -> Optional[Tuple[str, Optional[float]]]:
    if ts is None or ts.empty or "value" not in ts or "date" not in ts or len(ts) < 2:
        return None
    first = float(ts.iloc[0]["value"])
    last = float(ts.iloc[-1]["value"])
    direction = "increasing" if last > first else "decreasing" if last < first else "stable"
    pct = round(((last - first) / first) * 100, 1) if first != 0 else None
    return direction, pct


def find_extremes(ts: pd.DataFrame) -> Tuple[Optional[Tuple[datetime, float]], Optional[Tuple[datetime, float]]]:
    if ts is None or ts.empty:
        return None, None
    i_max = ts["value"].idxmax()
    i_min = ts["value"].idxmin()
    max_point = (pd.to_datetime(ts.loc[i_max, "date"]), float(ts.loc[i_max, "value"]))
    min_point = (pd.to_datetime(ts.loc[i_min, "date"]), float(ts.loc[i_min, "value"]))
    return max_point, min_point


def analyze_seasonality(ts: pd.DataFrame) -> Optional[Dict[int, float]]:
    if ts is None or ts.empty or len(ts) < 18:
        return None
    tmp = ts.copy()
    tmp["month"] = pd.to_datetime(tmp["date"]).dt.month
    m = tmp.groupby("month")["value"].mean().round(1)
    return {int(k): float(v) for k, v in m.to_dict().items()}


def forecast_linear(ts: pd.DataFrame, target_date: str) -> Optional[float]:
    if ts is None or ts.empty or len(ts) < 2:
        return None
    try:
        base = pd.to_datetime(ts["date"])
        x = (base - base.min()).dt.days.values
        y = ts["value"].astype(float).values
        coeffs = np.polyfit(x, y, 1)
        target_x = (pd.to_datetime(target_date) - base.min()).days
        pred = float(np.polyval(coeffs, target_x))
        return round(pred, 1)
    except Exception as e:
        logger.info(f"forecast failed: {e}")
        return None


# ---------------- LLM analysis pass ----------------
ANALYST_SYSTEM = """
You are an energy market analyst. Use ONLY the numbers given in the prompt.
Do NOT mention SQL, schema, tables, or columns. Write a clear analysis:
- Direction & approximate % change (first→last)
- Peaks/lows (when and how much)
- Seasonality if visible (hydro ↑ spring/summer; thermal often ↑ winter)
- If forecast implied by the question, include the projection result handed to you
- For multiple series, compare them succinctly
- End with a one-line takeaway
Be concise and avoid jargon.
"""

def llm_analyst_answer(llm: ChatOpenAI,
                       user_query: str,
                       unit: Optional[str],
                       series_dict: Dict[str, pd.DataFrame],
                       computed: Dict[str, Any]) -> str:
    """
    Give the LLM a compact, grounded brief: stats + a tiny sample of points.
    """
    # Build a compact data summary
    summary_lines = []
    for name, ts in series_dict.items():
        if "date" in ts.columns:
            # include only a few samples to ground the LLM, not the entire timeseries
            sample = ts.tail(6).copy()
            sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%Y-%m")
            pairs = [f"{d}:{round(float(v),1)}" for d, v in zip(sample["date"], sample["value"])]
            summary_lines.append(f"{name}: {', '.join(pairs)}")
        else:
            cats = ts.sort_values("value", ascending=False).head(6)
            pairs = [f"{str(l)}:{round(float(v),1)}" for l, v in zip(cats["label"], cats["value"])]
            summary_lines.append(f"{name} (top): {', '.join(pairs)}")

    computed_txt = []
    for k, v in computed.items():
        computed_txt.append(f"{k}: {v}")

    prompt = (
        f"User query: {user_query}\n"
        f"Unit: {unit or 'Value'}\n"
        f"Series preview:\n- " + "\n- ".join(summary_lines) + "\n"
        f"Computed stats:\n- " + "\n- ".join(computed_txt) + "\n"
        "Write the answer now."
    )

    msg = llm.invoke([
        {"role": "system", "content": ANALYST_SYSTEM},
        {"role": "user", "content": prompt}
    ])
    content = getattr(msg, "content", str(msg)) if msg else ""
    return content.strip()


# ---------------- Endpoint helpers ----------------
def extract_sql_from_steps(steps: List[Any]) -> Optional[str]:
    if not steps:
        return None
    # Steps often come as [(action_dict, observation), ...]
    for step in reversed(steps):
        try:
            if isinstance(step, tuple) and isinstance(step[0], dict) and "sql_cmd" in step[0]:
                return step[0]["sql_cmd"]
            if isinstance(step, dict) and "sql_cmd" in step:
                return step["sql_cmd"]
        except Exception:
            continue
    return None


def contains_future_intent(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["forecast", "predict", "expected", "future", "projection", "2030", "2035", "next year", "next 5 years"])


# ---------------- Endpoints ----------------
@app.get("/healthz")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "ts": datetime.utcnow()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=APIResponse)
def ask(q: Question, x_app_key: str = Header(...)):
    import time
    start = time.time()

    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # 1) Build memory-augmented context for better intent inference
        final_input = build_context(q.user_id, q.query)

        # 2) Create SQL agent (generation only)
        llm_agent = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY, request_timeout=45)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm_agent)
        agent = create_sql_agent(
            llm=llm_agent,
            toolkit=toolkit,
            verbose=True,
            agent_type="openai-tools",
            system_message=SYSTEM_PROMPT,
            max_iterations=6,
            early_stopping_method="generate",
        )

        # 3) Invoke agent with robust parsing
        try:
            result = agent.invoke({"input": final_input, "handle_parsing_errors": True}, return_intermediate_steps=True)
        except Exception as e:
            logger.error(f"Agent parsing failed: {e}")
            # Retry once without history
            result = agent.invoke({"input": q.query, "handle_parsing_errors": True}, return_intermediate_steps=True)

        output_text = result.get("output", "") or ""
        steps = result.get("intermediate_steps", []) or []

        # 4) Extract SQL and execute safely
        sql = extract_sql_from_steps(steps)
        if not sql:
            # No SQL found → return LLM text (scrubbed)
            ans = scrub_schema_mentions(output_text or "I don't know based on the available data.")
            return APIResponse(answer=ans, execution_time=round(time.time() - start, 2))

        sql = clean_sql(sql)
        validate_sql_is_safe(sql)

        with engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()

        if not rows:
            ans = scrub_schema_mentions("I don't have data for that specific request.")
            return APIResponse(answer=ans, execution_time=round(time.time() - start, 2))

        df = coerce_dataframe(rows)

        # If we got very few rows and it looks like time series, try expanding LIMIT
        if detect_timeseries(df) and df.shape[0] < 24:
            expanded_sql = try_expand_limit_for_timeseries(sql, df)
            if expanded_sql != sql:
                with engine.connect() as conn:
                    rows2 = conn.execute(text(expanded_sql)).fetchall()
                if rows2:
                    df2 = coerce_dataframe(rows2)
                    if df2.shape[0] > df.shape[0]:
                        df = df2
                        sql = expanded_sql

        # 5) Build one or more series
        series_dict = extract_series(df)
        if not series_dict:
            ans = scrub_schema_mentions(output_text or "I don't know based on the available data.")
            return APIResponse(answer=ans, execution_time=round(time.time() - start, 2))

        unit = infer_unit_from_query(q.query)

        # 6) Compute numeric stats per series
        computed: Dict[str, Any] = {}
        # Per-series analytics
        for name, s in series_dict.items():
            if "date" in s.columns:
                s = s.dropna()
                if s.empty:
                    continue
                tr = analyze_trend(s)
                if tr:
                    direction, pct = tr
                    computed[f"{name}__trend"] = f"{direction} ({pct:+.1f}%)" if pct is not None else direction
                mx, mn = find_extremes(s)
                if mx:
                    computed[f"{name}__peak"] = f"{mx[0].strftime('%Y-%m')}: {round(mx[1],1)}"
                if mn:
                    computed[f"{name}__low"] = f"{mn[0].strftime('%Y-%m')}: {round(mn[1],1)}"
                seas = analyze_seasonality(s)
                if seas:
                    computed[f"{name}__seasonality_hint"] = "seasonality visible"
                # Forecast if implied
                if contains_future_intent(q.query):
                    # pick "2030-12-01" as canonical endpoint if "2030" appears, else use +5y
                    target = "2030-12-01" if "2030" in q.query else (pd.to_datetime(s["date"]).max() + pd.DateOffset(years=5)).strftime("%Y-%m-%d")
                    pred = forecast_linear(s, target)
                    if pred is not None:
                        computed[f"{name}__forecast_{target}"] = pred
                series_dict[name] = s  # store back (cleaned)
            else:
                # categorical: total & top
                total = float(s["value"].sum())
                computed[f"{name}__total"] = round(total, 1)
                if not s.empty:
                    top_row = s.sort_values("value", ascending=False).iloc[0]
                    share = (float(top_row["value"]) / total * 100) if total > 0 else 0
                    computed[f"{name}__top"] = f"{str(top_row['label'])}: {round(float(top_row['value']),1)} ({share:.1f}% share)"

        # 7) LLM analysis pass (like me 😄): interpret & write
        llm_analyst = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_API_KEY, request_timeout=45)
        final_text = llm_analyst_answer(llm_analyst, q.query, unit, series_dict, computed)

        # Fallback if somehow empty
        if not final_text.strip():
            # Build deterministic numeric summary
            lines = []
            for k in sorted(computed.keys()):
                lines.append(f"{k}: {computed[k]}")
            final_text = "Here is a numeric summary based on the available data:\n" + "\n".join(lines)

        final_text = scrub_schema_mentions(final_text)

        return APIResponse(answer=final_text, execution_time=round(time.time() - start, 2))

    except SQLAlchemyError as e:
        logger.error(f"DB error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail="Bad request")
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Processing error")
