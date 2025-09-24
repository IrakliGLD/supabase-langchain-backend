import os
import re
import logging
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
from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd

# Internal schema doc
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
    "trade"
]

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
app = FastAPI(title="EnerBot Backend", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- System Prompt ----------------
SYSTEM_PROMPT = f"""
You are EnerBot, an autonomous Georgian electricity market analyst.

=== RULES ===
- ONLY use numbers from the database queries (no outside data, no guesses).
- NEVER reveal SQL, schema, table names, or column names.
- Always analyze, not just list numbers.
- Provide: trend direction, % changes, peaks/lows, anomalies, seasonality (if relevant).
- If forecast/predict/future → extrapolate trend.
- End with a one-line key insight.
- Always answer in plain language with units.

=== INTERNAL SCHEMA (hidden from user) ===
{DB_SCHEMA_DOC}
"""

# ---------------- Models ----------------
class Question(BaseModel):
    query: str = Field(..., max_length=1000)
    user_id: Optional[str] = None

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class APIResponse(BaseModel):
    answer: str
    execution_time: Optional[float] = None

# ---------------- Utils ----------------
def scrub_schema_mentions(text: str) -> str:
    if not text:
        return text
    replacements = {
        "p_bal_gel": "balancing electricity price (GEL)",
        "p_dereg_gel": "deregulated electricity price (GEL)",
        "p_gcap_gel": "guaranteed capacity price (GEL)",
        "tariff_gel": "tariff (GEL)",
        "volume_tj": "energy volume (TJ)",
        "quantity_tech": "technology quantity",
        "xrate": "exchange rate",
    }
    for k, v in replacements.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text, flags=re.IGNORECASE)
    for t in ALLOWED_TABLES:
        text = re.sub(rf"\b{re.escape(t)}\b", "the database", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(schema|table|column|sql|join)\b", "data", text, flags=re.IGNORECASE)
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

def prepare_timeseries_data(df: pd.DataFrame):
    if df.shape[1] < 2:
        return None, False
    df_clean = df.copy()
    df_clean.columns = ["key", "value"]
    df_clean["value"] = pd.to_numeric(df_clean["value"], errors="coerce")
    df_clean = df_clean.dropna()
    try:
        df_clean["date"] = pd.to_datetime(df_clean["key"])
        return df_clean[["date", "value"]].sort_values("date"), True
    except Exception:
        return df_clean.rename(columns={"key": "label"}), False

def analyze_trend(ts: pd.DataFrame):
    if len(ts) < 2:
        return None
    first, last = ts.iloc[0]["value"], ts.iloc[-1]["value"]
    direction = "increasing" if last > first else "decreasing" if last < first else "stable"
    pct = round((last - first) / first * 100, 1) if first != 0 else None
    return direction, pct

def forecast_linear(ts: pd.DataFrame, target_date="2030-12-01"):
    if len(ts) < 2:
        return None
    x = (ts["date"] - ts["date"].min()).dt.days.values
    y = ts["value"].values
    coeffs = np.polyfit(x, y, 1)
    target = (pd.to_datetime(target_date) - ts["date"].min()).days
    return round(float(np.polyval(coeffs, target)), 1)

# ---------------- Memory (last 3 turns) ----------------
def get_recent_history(user_id: str, limit=3):
    if not user_id:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT role, content FROM chat_history WHERE user_id = :u ORDER BY created_at DESC LIMIT :l"),
                {"u": user_id, "l": limit * 2},
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception:
        return []

def build_context(user_id: str, query: str):
    history = get_recent_history(user_id, 3)
    if not history:
        return query
    history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
    return f"{history_text}\nUser: {query}\nAssistant:"

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
        # Build context with memory
        final_input = build_context(q.user_id, q.query) if q.user_id else q.query

        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            system_message=SYSTEM_PROMPT,
            verbose=True,
            handle_parsing_errors=True   # 👈 tolerate bad LLM outputs
        )

        try:
            result = agent.invoke({"input": final_input}, return_intermediate_steps=True)
        except Exception as e:
            logger.error(f"Agent parsing failed: {e}")
            return APIResponse(
                answer="I don't know based on the available data.",
                execution_time=round(time.time() - start, 2)
            )

        output = result.get("output", "")
        steps = result.get("intermediate_steps", [])

        # Try to extract SQL
        sql = None
        for step in reversed(steps):
            if isinstance(step, tuple) and isinstance(step[0], dict) and "sql_cmd" in step[0]:
                sql = step[0]["sql_cmd"]
                break
            if isinstance(step, dict) and "sql_cmd" in step:
                sql = step["sql_cmd"]
                break

        if sql:
            with engine.connect() as conn:
                rows = conn.execute(text(sql)).fetchall()
            df = pd.DataFrame(rows)
            if not df.empty:
                ts, is_ts = prepare_timeseries_data(df)
                if is_ts:
                    trend = analyze_trend(ts)
                    fc = forecast_linear(ts) if any(k in q.query.lower() for k in ["forecast", "predict", "2030"]) else None
                    analysis = []
                    if trend:
                        direction, pct = trend
                        analysis.append(f"Trend: {direction} ({pct}% change overall).")
                    if fc:
                        analysis.append(f"Projection for Dec 2030: {fc}.")
                    if analysis:
                        return APIResponse(
                            answer=scrub_schema_mentions("\n".join(analysis)),
                            execution_time=round(time.time()-start,2)
                        )

        return APIResponse(answer=scrub_schema_mentions(output), execution_time=round(time.time()-start,2))

    except Exception as e:
        logger.error(f"Ask failed: {e}")
        return APIResponse(
            answer="I encountered an error while processing your request. Please try again.",
            execution_time=round(time.time() - start, 2)
        )
