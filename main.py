import os
import re
import logging
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
app = FastAPI(title="EnerBot Backend", version="4.3-seasonal-ml")

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

=== MANDATORY BEHAVIOR ===
- Use only the numbers you query from the database. No outside sources or guesses.
- NEVER reveal SQL, schema, table names, or column names in your final answer.
- Always analyze instead of listing: trend direction, % change (first→last), peaks/lows, anomalies, and seasonality if the pattern exists.
- If the user implies a forecast or “if the current trend maintains…”, compute a forward-looking projection from the available data.
- Always answer in plain language with units when applicable and end with a one-line key insight.

=== DATASET SELECTION (based on the internal schema only) ===
(1) Choose the dataset solely from the schema descriptions in the internal context:
    • Generation by technology (hydro, thermal, wind, solar, etc.) → technology time-series dataset.
    • Consumption by sector/fuel → sector balance dataset.
    • Prices (deregulated, balancing, guaranteed capacity) → prices dataset.
    • Tariffs → tariffs dataset.
(2) Do NOT use tiny samples for time-based questions. Avoid small LIMITs (e.g., LIMIT 10) when analyzing multi-year trends. Query the full range unless a sample is explicitly requested.
(3) If the user references thermal/hydro (or synonyms like HPP/TPP), prefer the technology-based generation dataset for those technologies.

=== TERMINOLOGY ===
The user may use natural or industry terms (e.g., HPP/Hydro power plant → hydro; TPP/Thermal power plant → thermal).
Interpret terms naturally using the schema documentation and select the correct dataset without asking for confirmation.

=== INTERNAL SCHEMA (hidden from user) ===
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


# ---------------- Scrubber (no schema leakage) ----------------
def scrub_schema_mentions(text: str) -> str:
    """Remove table/schema/SQL jargon from final output."""
    if not text:
        return text
    # Replace allowed table names with neutral phrase
    for t in ALLOWED_TABLES:
        text = re.sub(rf"\b{re.escape(t)}\b", "the database", text, flags=re.IGNORECASE)
    # Replace database jargon
    text = re.sub(r"\b(schema|table|column|sql|join|primary key|foreign key|view|constraint)\b", "data", text, flags=re.IGNORECASE)
    # Clean repeated phrase
    text = re.sub(r"(the database\s+){2,}", "the database ", text)
    # Remove stray backticks
    text = text.replace("```", "").strip()
    return text


# ---------------- Helpers ----------------
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
def clean_sql(sql: str) -> str:
    if not sql:
        return sql
    sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```\s*", "", sql)
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = sql.strip()
    if sql.endswith(";"):
        sql = sql[:-1]
    # Strip LIMIT completely to force full-series retrieval for analysis
    sql = re.sub(r"\bLIMIT\s+\d+\b", "", sql, flags=re.IGNORECASE)
    return sql.strip()


def validate_sql_is_safe(sql: str) -> None:
    up = sql.upper()
    if not up.startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed.")


# ---------------- Data shaping ----------------
def coerce_dataframe(rows: List[tuple]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([list(r) for r in rows])
    df = df.applymap(convert_decimal_to_float)
    return df


def detect_timeseries(df: pd.DataFrame) -> bool:
    if df.empty or df.shape[1] < 2:
        return False
    for i in range(df.shape[1]):
        try:
            pd.to_datetime(df.iloc[:, i])
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
    Return dict of named series.
    If a categorical column exists alongside date+value, split by category.
    """
    out: Dict[str, pd.DataFrame] = {}

    if df.empty:
        return out

    # Find a datetime-like column
    date_col_idx = None
    for i in range(df.shape[1]):
        try:
            pd.to_datetime(df.iloc[:, i])
            date_col_idx = i
            break
        except Exception:
            continue

    if date_col_idx is not None:
        date_col = pd.to_datetime(df.iloc[:, date_col_idx], errors="coerce")

        # Candidate numeric cols
        numeric_idx = [j for j in range(df.shape[1]) if j != date_col_idx and pd.api.types.is_numeric_dtype(df.iloc[:, j])]

        # Candidate category col (first non-numeric, non-date)
        cat_idx = None
        for j in range(df.shape[1]):
            if j == date_col_idx:
                continue
            if not pd.api.types.is_numeric_dtype(df.iloc[:, j]):
                cat_idx = j
                break

        if numeric_idx and cat_idx is not None:
            val_col = numeric_idx[0]
            cats = df.iloc[:, cat_idx].astype(str)
            vals = pd.to_numeric(df.iloc[:, val_col], errors="coerce")
            tmp = pd.DataFrame({"date": date_col, "cat": cats, "value": vals}).dropna()
            for label, g in tmp.groupby("cat"):
                s = g[["date", "value"]].sort_values("date").reset_index(drop=True)
                out[str(label)] = s
        elif numeric_idx:
            val_col = numeric_idx[0]
            vals = pd.to_numeric(df.iloc[:, val_col], errors="coerce")
            s = pd.DataFrame({"date": date_col, "value": vals}).dropna().sort_values("date").reset_index(drop=True)
            out["series"] = s
        else:
            return {}
    else:
        # Fallback: categories + values (no date)
        if df.shape[1] < 2:
            return {}
        lbl = df.iloc[:, 0].astype(str)
        vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s = pd.DataFrame({"label": lbl, "value": vals}).dropna().reset_index(drop=True)
        out["categories"] = s

    return out


# ---------------- Numeric analytics (model-driven seasonality support) ----------------
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


def compute_monthly_profile(ts: pd.DataFrame) -> Optional[Dict[int, float]]:
    """Return monthly average profile {1..12: mean} if time series is long enough."""
    if ts is None or ts.empty or "date" not in ts or "value" not in ts:
        return None
    # Need at least ~18 points to talk about seasonality credibly
    if len(ts) < 18:
        return None
    tmp = ts.copy()
    tmp["month"] = pd.to_datetime(tmp["date"]).dt.month
    monthly = tmp.groupby("month")["value"].mean().round(1)
    if monthly.empty:
        return None
    return {int(m): float(v) for m, v in monthly.to_dict().items()}


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


def contains_future_intent(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["forecast", "predict", "expected", "future", "projection", "2030", "2035", "next year", "next 5 years", "maintains"])


# ---------------- LLM analysis pass ----------------
ANALYST_SYSTEM = """
You are an energy market analyst. Use ONLY the numbers provided below (they come from the database).
Do NOT mention SQL, schema, tables, or columns. Write a clear, concise analysis:
- Direction & approximate % change from first to last point.
- Peaks and lows (when and how much).
- Seasonality: infer only from the monthly profile given (if present). You may note typical domain patterns,
  e.g., hydro often peaks in spring/summer and thermal in winter, BUT anchor your comments in the provided monthly profile.
- If a forecast is implied, include the provided projection result(s) and caveat that it’s a simple trend extrapolation.
- For multiple series (e.g., hydro vs thermal), compare them succinctly.
- End with one short takeaway in plain language.
Avoid jargon. Keep it grounded in the data below.
"""

def llm_analyst_answer(
    llm: ChatOpenAI,
    user_query: str,
    unit: Optional[str],
    series_dict: Dict[str, pd.DataFrame],
    computed: Dict[str, Any],
    monthly_profiles: Dict[str, Optional[Dict[int, float]]]
) -> str:
    # Build compact previews for the model
    lines = []
    for name, ts in series_dict.items():
        if "date" in ts.columns:
            sample = ts.tail(6).copy()
            sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%Y-%m")
            pairs = [f"{d}:{round(float(v),1)}" for d, v in zip(sample["date"], sample["value"])]
            mp = monthly_profiles.get(name)
            mp_str = f" | monthly_profile={mp}" if mp else ""
            lines.append(f"{name}: " + ", ".join(pairs) + mp_str)
        else:
            cats = ts.sort_values("value", ascending=False).head(6)
            pairs = [f"{str(l)}:{round(float(v),1)}" for l, v in zip(cats["label"], cats["value"])]
            lines.append(f"{name} (top): {', '.join(pairs)}")

    stats = [f"{k}: {v}" for k, v in computed.items()]
    prompt = (
        f"User query: {user_query}\n"
        f"Unit: {unit or 'Value'}\n"
        f"Series preview (last points & monthly profiles if present):\n- " + "\n- ".join(lines) + "\n"
        f"Computed stats:\n- " + "\n- ".join(stats) + "\n"
        "Write the answer now."
    )

    msg = llm.invoke([
        {"role": "system", "content": ANALYST_SYSTEM},
        {"role": "user", "content": prompt}
    ])
    content = getattr(msg, "content", str(msg)) if msg else ""
    return content.strip()


# ---------------- Helpers to extract SQL ----------------
def extract_sql_from_steps(steps: List[Any]) -> Optional[str]:
    """
    Robustly extract SQL from LangChain agent steps.
    Supports:
      - action["sql_cmd"]
      - action["tool_input"]["query"]
      - action["tool_input"] as a raw SQL string
    Prefers the *last* sql_db_query or sql_db_query_checker step.
    """
    if not steps:
        return None

    candidate_sql = None

    for step in steps:
        try:
            action = step[0] if isinstance(step, tuple) and step else step
            if not isinstance(action, dict):
                continue

            # direct sql_cmd
            if "sql_cmd" in action and isinstance(action["sql_cmd"], str):
                candidate_sql = action["sql_cmd"]

            # tool_input might hold SQL
            tool_input = action.get("tool_input")
            if isinstance(tool_input, dict) and isinstance(tool_input.get("query"), str):
                candidate_sql = tool_input["query"]
            if isinstance(tool_input, str) and tool_input.strip().upper().startswith("SELECT"):
                candidate_sql = tool_input
        except Exception:
            continue

    return candidate_sql


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
        # 1) Build memory-augmented context
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
            # retry with raw query only
            result = agent.invoke({"input": q.query, "handle_parsing_errors": True}, return_intermediate_steps=True)

        output_text = result.get("output", "") or ""
        steps = result.get("intermediate_steps", []) or []

        # 4) Extract SQL and execute safely
        sql = extract_sql_from_steps(steps)
        if not sql:
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

        # If it does not look like a time series, just let the LLM summarize numbers
        looks_ts = detect_timeseries(df)
        series_dict = extract_series(df)
        if not series_dict:
            ans = scrub_schema_mentions(output_text or "I don't know based on the available data.")
            return APIResponse(answer=ans, execution_time=round(time.time() - start, 2))

        unit = infer_unit_from_query(q.query)

        # 5) Compute analytics per series (trend/extremes/forecast + model-driven seasonality)
        computed: Dict[str, Any] = {}
        monthly_profiles: Dict[str, Optional[Dict[int, float]]] = {}

        for name, s in series_dict.items():
            if "date" in s.columns:
                s = s.dropna()
                if s.empty:
                    continue
                # trend
                tr = analyze_trend(s)
                if tr:
                    direction, pct = tr
                    computed[f"{name}__trend"] = f"{direction} ({pct:+.1f}%)" if pct is not None else direction
                # extremes
                mx, mn = find_extremes(s)
                if mx:
                    computed[f"{name}__peak"] = f"{mx[0].strftime('%Y-%m')}: {round(mx[1],1)}"
                if mn:
                    computed[f"{name}__low"] = f"{mn[0].strftime('%Y-%m')}: {round(mn[1],1)}"
                # monthly profile (seasonality input for the LLM)
                mp = compute_monthly_profile(s)
                if mp:
                    monthly_profiles[name] = mp
                else:
                    monthly_profiles[name] = None
                # forecast if implied
                if contains_future_intent(q.query):
                    target = "2030-12-01" if "2030" in q.query else (pd.to_datetime(s["date"]).max() + pd.DateOffset(years=5)).strftime("%Y-%m-%d")
                    pred = forecast_linear(s, target)
                    if pred is not None:
                        computed[f"{name}__forecast_{target}"] = pred
                series_dict[name] = s
            else:
                # categories-only (no date)
                total = float(s["value"].sum())
                computed[f"{name}__total"] = round(total, 1)
                if not s.empty:
                    top_row = s.sort_values("value", ascending=False).iloc[0]
                    share = (float(top_row["value"]) / total * 100) if total > 0 else 0
                    computed[f"{name}__top"] = f"{str(top_row['label'])}: {round(float(top_row['value']),1)} ({share:.1f}% share)"

        # 6) LLM analysis pass (final narrative)
        llm_analyst = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_API_KEY, request_timeout=45)
        final_text = llm_analyst_answer(llm_analyst, q.query, unit, series_dict, computed, monthly_profiles)

        if not final_text.strip():
            lines = [f"{k}: {v}" for k, v in sorted(computed.items())]
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
