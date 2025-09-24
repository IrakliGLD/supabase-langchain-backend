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
from langchain.agents import create_sql_agent, AgentExecutor
from langchain_core.agents import AgentAction
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import Tool
from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
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

# ---------------- Database & Join Knowledge Graph ----------------
ALLOWED_TABLES = [
    "energy_balance_long",
    "entities",
    "monthly_cpi",
    "price",
    "tariff_gen",
    "tech_quantity",
    "trade",
]

DB_JOINS = {
    "energy_balance_long": {"join_on": "date", "related_to": ["price", "trade"]},
    "price": {"join_on": "date", "related_to": ["energy_balance_long"]},
    "trade": {"join_on": "date", "related_to": ["energy_balance_long"]},
    "tech_quantity": {"join_on": "date", "related_to": []},
    "entities": {"join_on": "id", "related_to": []},
    "monthly_cpi": {"join_on": "date", "related_to": []},
    "tariff_gen": {"join_on": "date", "related_to": []},
}

engine = create_engine(
    SUPABASE_DB_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)
db = SQLDatabase(engine, include_tables=ALLOWED_TABLES)

# ---------------- FastAPI ----------------
app = FastAPI(title="EnerBot Backend", version="5.1-resilient-analyst-v2")

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

You MUST follow this EXACT process step-by-step for EVERY query:
1. Determine the user's intent: Is it for a simple value, summary statistic, trend analysis, or future forecast?
2. Identify the required columns and tables based on the user's query and the internal schema.
3. If the query requires combining data from multiple tables, use the `DB_JOINS` knowledge provided below to find the correct `JOIN` column.
4. If the query is for **trend analysis, seasonality checks, or forecasts**, you MUST use the `sql_db_query_full_series` tool. This tool will automatically query the full dataset without any limits.
5. If the query is for a **specific value or a small sample**, you may use the standard `sql_db_query` tool.
6. Execute the final query and generate an intermediate response.

=== MANDATORY BEHAVIOR ===
- Use only the numbers you query from the database. No outside sources or guesses.
- NEVER reveal SQL, schema, table names, or column names in your final answer.
- Your final output must be an analysis, not a list of numbers. Include trends, % changes, peaks, lows, and seasonality.
- End with one short key insight.
- Answer in plain language with units (TJ, GEL, etc.) when applicable.

=== DATASET SELECTION & JOINS ===
(1) Generation by technology (hydro, thermal, wind, solar) → `tech_quantity`
(2) Consumption by sector/fuel → `energy_balance_long`
(3) Prices → `price`
(4) Trade (imports, exports) → `trade`
(5) To join these tables, use the `date` column. The `DB_JOINS` dictionary below confirms these relationships.

=== TERMINOLOGY ===
The user may use natural or industry terms (e.g., HPP/Hydro power plant → hydro; TPP/Thermal power plant → thermal). Interpret terms naturally using the schema documentation and select the correct dataset without asking for confirmation.

=== INTERNAL SCHEMA (hidden from user) ===
{DB_SCHEMA_DOC}

=== DB JOIN KNOWLEDGE ===
DB_JOINS = {DB_JOINS}
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
    if not text:
        return text
    for t in ALLOWED_TABLES:
        text = re.sub(rf"\b{re.escape(t)}\b", "the database", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(schema|table|column|sql|join|primary key|foreign key|view|constraint)\b", "data", text, flags=re.IGNORECASE)
    text = re.sub(r"(the database\s+){2,}", "the database ", text)
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
    sql = re.sub(r"```(?:sql)?\s*|\s*```", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = sql.strip()
    if sql.endswith(";"):
        sql = sql[:-1]
    sql = re.sub(r"\bLIMIT\s+\d+\b", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bOFFSET\s+\d+\b", "", sql, flags=re.IGNORECASE)
    return sql.strip()

def validate_sql_is_safe(sql: str) -> None:
    up = sql.upper()
    if not up.startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed.")
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]
    if any(f in up for f in forbidden):
        raise ValueError("Only SELECT statements are allowed.")

# ---------------- Data shaping ----------------
def coerce_dataframe(rows: List[tuple]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([list(r) for r in rows])
    df = df.map(convert_decimal_to_float)
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
    out: Dict[str, pd.DataFrame] = {}
    if df.empty:
        return out
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
        numeric_idx = [j for j in range(df.shape[1]) if j != date_col_idx and pd.api.types.is_numeric_dtype(df.iloc[:, j])]
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

def find_anomalies(ts: pd.DataFrame, threshold: float = 3.0) -> List[Tuple[datetime, float]]:
    if ts is None or ts.empty or len(ts) < 3:
        return []
    mean = ts["value"].mean()
    std = ts["value"].std()
    if std == 0:
        return []
    z = np.abs((ts["value"] - mean) / std)
    outliers = ts[z > threshold]
    return [(pd.to_datetime(row["date"]), float(row["value"])) for _, row in outliers.iterrows()]

def compute_seasonality(ts: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if ts is None or ts.empty or "date" not in ts or "value" not in ts or len(ts) < 24:
        tmp = ts.copy()
        tmp["month"] = pd.to_datetime(tmp["date"]).dt.month
        monthly = tmp.groupby("month")["value"].mean().round(1)
        if monthly.empty:
            return None
        return {"monthly_profile": {int(m): float(v) for m, v in monthly.to_dict().items()}}
    try:
        ts_copy = ts.set_index("date").sort_index()
        ts_copy = ts_copy.asfreq("M", method="ffill")
        decomp = sm.tsa.seasonal_decompose(ts_copy["value"], model="additive", period=12)
        seasonal = decomp.seasonal.dropna().round(1)
        return {"seasonal_component": seasonal.to_dict()}
    except Exception as e:
        logger.info(f"Seasonality decomposition failed: {e}")
        return None

def forecast_arima(ts: pd.DataFrame, steps: int = 12) -> Optional[Dict[str, float]]:
    if ts is None or ts.empty or len(ts) < 36:
        return None
    try:
        ts_copy = ts.set_index("date").sort_index().asfreq("M", method="ffill")
        model = ARIMA(ts_copy["value"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
        forecast = model.forecast(steps=steps)
        out = {pd.to_datetime(d).strftime("%Y-%m"): float(v) for d, v in zip(forecast.index, forecast.values)}
        return out
    except Exception as e:
        logger.info(f"ARIMA forecast failed: {e}")
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
- Anomalies (outliers) if any.
- Seasonality: infer only from the provided seasonal component or monthly profile.
- If a forecast is provided, include the projection results and caveat that it's a model-based extrapolation.
- For multiple series, compare them succinctly.
- End with one short takeaway in plain language.
Avoid jargon. Keep it grounded in the data below.
"""

def llm_analyst_answer(
    llm: ChatOpenAI,
    user_query: str,
    unit: Optional[str],
    series_dict: Dict[str, pd.DataFrame],
    computed: Dict[str, Any],
    seasonality_info: Dict[str, Optional[Dict[str, Any]]]
) -> str:
    lines = []
    for name, ts in series_dict.items():
        if "date" in ts.columns:
            sample = ts.tail(6).copy()
            sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%Y-%m")
            pairs = [f"{d}:{round(float(v),1)}" for d, v in zip(sample["date"], sample["value"])]
            si = seasonality_info.get(name)
            si_str = f" | seasonality={si}" if si else ""
            lines.append(f"{name}: " + ", ".join(pairs) + si_str)
        else:
            cats = ts.sort_values("value", ascending=False).head(6)
            pairs = [f"{str(l)}:{round(float(v),1)}" for l, v in zip(cats["label"], cats["value"])]
            lines.append(f"{name} (top): {', '.join(pairs)}")
    stats = [f"{k}: {v}" for k, v in computed.items()]
    prompt = (
        f"User query: {user_query}\n"
        f"Unit: {unit or 'Value'}\n"
        f"Series preview (last points & seasonality if present):\n- " + "\n- ".join(lines) + "\n"
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
    if not steps:
        return None
    for step in reversed(steps):
        try:
            if isinstance(step, tuple) and len(step) >= 2:
                action, observation = step[0], step[1]
            else:
                action = step
                observation = ""
            action_dict = action.dict() if isinstance(action, AgentAction) else action
            if not isinstance(action_dict, dict):
                continue
            tool = action_dict.get("tool", "")
            if tool not in ["sql_db_query", "sql_db_query_checker", "sql_db_query_full_series"]:
                continue
            if "sql_cmd" in action_dict and isinstance(action_dict["sql_cmd"], str):
                return action_dict["sql_cmd"]
            tool_input = action_dict.get("tool_input")
            if isinstance(tool_input, dict) and isinstance(tool_input.get("query"), str):
                return tool_input["query"]
            if isinstance(tool_input, str) and tool_input.strip().upper().startswith("SELECT"):
                return tool_input
            log = action_dict.get("log", "")
            if "SELECT" in log.upper():
                match = re.search(r"SELECT.*?($|;)", log, re.DOTALL | re.IGNORECASE)
                if match:
                    return match.group(0).strip()
            if observation and isinstance(observation, str) and "SELECT" in observation.upper():
                match = re.search(r"SELECT.*?($|;)", observation, re.DOTALL | re.IGNORECASE)
                if match:
                    return match.group(0).strip()
        except Exception as e:
            logger.debug(f"Failed to extract SQL from step: {e}")
            continue
    return None

# ---------------- Endpoints ----------------
@app.get("/healthz")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "ts": datetime.utcnow()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ... (rest of the code is the same) ...

@app.post("/ask", response_model=APIResponse)
def ask(q: Question, x_app_key: str = Header(...)):
    import time
    start = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        final_input = build_context(q.user_id, q.query)
        llm_agent = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY, request_timeout=60)
        
        # Define a custom tool for full-series queries
        def run_full_series_query(query: str) -> str:
            cleaned_query = clean_sql(query)
            validate_sql_is_safe(cleaned_query)
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(cleaned_query)).fetchall()
                    return str(result)
            except Exception as e:
                return f"Error executing query: {e}"

        custom_tools = [
            Tool(
                name="sql_db_query_full_series",
                func=run_full_series_query,
                description="Use this tool ONLY for trend analysis, seasonality, or forecasting. It will execute a query and retrieve ALL results without a LIMIT clause."
            )
        ]

        toolkit = SQLDatabaseToolkit(db=db, llm=llm_agent)
        
        # Get all default tools except the one we want to replace
        all_tools = [t for t in toolkit.get_tools() if t.name not in ["sql_db_query", "sql_db_query_checker"]]

        # Add the custom tool
        all_tools.extend(custom_tools)

        agent = create_sql_agent(
            llm=llm_agent,
            toolkit=toolkit,
            tools=all_tools, # Use the new, filtered list of tools
            verbose=True,
            agent_type="openai-tools",
            system_message=SYSTEM_PROMPT,
            max_iterations=8,
            early_stopping_method="generate",
        )
        
        # ... (rest of the try-except block is the same) ...
        
        try:
            result = agent.invoke({"input": final_input, "handle_parsing_errors": True}, return_intermediate_steps=True)
        except Exception as e:
            logger.error(f"Agent parsing failed, retrying with raw query only: {e}")
            result = agent.invoke({"input": q.query, "handle_parsing_errors": True}, return_intermediate_steps=True)

        output_text = result.get("output", "") or ""
        steps = result.get("intermediate_steps", []) or []
        sql = extract_sql_from_steps(steps)
        
        if not sql:
            logger.warning(f"No SQL extracted from steps, falling back to agent output: {output_text}")
            ans = scrub_schema_mentions(output_text or "I don't know based on the available data.")
            return APIResponse(answer=ans, execution_time=round(time.time() - start, 2))
        
        # We still need to run the full query logic since the agent's observation might be truncated
        # and we need to pass the full dataframe to the analytics functions.
        clean_sql_query = clean_sql(sql)
        validate_sql_is_safe(clean_sql_query)
        
        with engine.connect() as conn:
            rows = conn.execute(text(clean_sql_query)).fetchall()
        
        if not rows:
            ans = scrub_schema_mentions("I don't have data for that specific request.")
            return APIResponse(answer=ans, execution_time=round(time.time() - start, 2))

        df = coerce_dataframe(rows)
        series_dict = extract_series(df)
        if not series_dict:
            ans = scrub_schema_mentions(output_text or "I don't know based on the available data.")
            return APIResponse(answer=ans, execution_time=round(time.time() - start, 2))

        unit = infer_unit_from_query(q.query)
        computed: Dict[str, Any] = {}
        seasonality_info: Dict[str, Optional[Dict[str, Any]]] = {}

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
                anomalies = find_anomalies(s)
                if anomalies:
                    computed[f"{name}__anomalies"] = [f"{d.strftime('%Y-%m')}: {round(v,1)}" for d, v in anomalies]
                si = compute_seasonality(s)
                if si:
                    seasonality_info[name] = si
                else:
                    seasonality_info[name] = None
                if contains_future_intent(q.query):
                    pred = forecast_arima(s)
                    if pred is not None:
                        computed[f"{name}__forecast"] = pred
                series_dict[name] = s
            else:
                total = float(s["value"].sum())
                computed[f"{name}__total"] = round(total, 1)
                if not s.empty:
                    top_row = s.sort_values("value", ascending=False).iloc[0]
                    share = (float(top_row["value"]) / total * 100) if total > 0 else 0
                    computed[f"{name}__top"] = f"{str(top_row['label'])}: {round(float(top_row['value']),1)} ({share:.1f}% share)"

        llm_analyst = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_API_KEY, request_timeout=60)
        final_text = llm_analyst_answer(llm_analyst, q.query, unit, series_dict, computed, seasonality_info)
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
