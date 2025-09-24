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
app = FastAPI(title="EnerBot Backend", version="4.2")

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

=== RULES ===
- ONLY use numbers from the database queries (no outside data, no guesses).
- NEVER reveal SQL, schema, table names, or column names.
- Always analyze, not just list numbers.
- Select the correct dataset based on the schema descriptions in the internal context. 
  * For generation by technology (hydro, thermal, wind, solar, etc.), always use the technology-based dataset. 
  * For consumption by sectors and fuels, use the sector balance dataset.
  * For prices, use the pricing dataset.
  * For tariffs, use the tariff dataset.
- User terminology may differ (e.g. "TPP", "Thermal power plant" → thermal; "HPP", "Hydro power plant" → hydro).
  You must interpret terms naturally, relying on schema descriptions to map them correctly.
- Do NOT use small LIMITs like LIMIT 10 when analyzing long-term trends.
  Always query the full range of available data unless explicitly asked for a sample.
- Provide: trend direction, % changes, peaks/lows, anomalies, seasonality (if relevant).
- If forecast/predict/future → extrapolate trend.
- End with a one-line key insight.
- Always answer in plain language with units.

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

    for k, v in TECH_TERM_MAP.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text, flags=re.IGNORECASE)

    for t in ALLOWED_TABLES:
        text = re.sub(rf"\b{re.escape(t)}\b", "the database", text, flags=re.IGNORECASE)

    text = re.sub(r"\b(schema|table|column|sql|join|primary key|foreign key)\b", "data", text, flags=re.IGNORECASE)
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
    # Strip LIMIT completely (Option 2 safeguard)
    sql = re.sub(r"\bLIMIT\s+\d+\b", "", sql, flags=re.IGNORECASE)
    return sql.strip()


def validate_sql_is_safe(sql: str) -> None:
    up = sql.upper()
    if not up.startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed.")


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
        final_input = build_context(q.user_id, q.query)

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

        result = agent.invoke({"input": final_input, "handle_parsing_errors": True}, return_intermediate_steps=True)

        output_text = result.get("output", "") or ""
        steps = result.get("intermediate_steps", []) or []

        sql = None
        for step in steps:
            if isinstance(step, tuple) and isinstance(step[0], dict) and "sql_cmd" in step[0]:
                sql = step[0]["sql_cmd"]
                break
            if isinstance(step, dict) and "sql_cmd" in step:
                sql = step["sql_cmd"]
                break

        if sql:
            sql = clean_sql(sql)
            validate_sql_is_safe(sql)
            with engine.connect() as conn:
                rows = conn.execute(text(sql)).fetchall()
            if not rows:
                return APIResponse(answer="I don't have data for that request.", execution_time=round(time.time()-start, 2))
            df = pd.DataFrame(rows)
            if not df.empty:
                # Simple trend analysis
                try:
                    df.columns = ["date", "category", "value"] if df.shape[1] == 3 else ["date", "value"]
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.dropna()
                    if not df.empty:
                        first, last = df.iloc[0]["value"], df.iloc[-1]["value"]
                        direction = "increasing" if last > first else "decreasing" if last < first else "stable"
                        pct = round((last - first) / first * 100, 1) if first != 0 else None
                        analysis = f"Trend: {direction} ({pct:+.1f}%)" if pct is not None else f"Trend: {direction}"
                        return APIResponse(answer=scrub_schema_mentions(analysis), execution_time=round(time.time()-start, 2))
                except Exception:
                    pass

        return APIResponse(answer=scrub_schema_mentions(output_text or "I don't know."), execution_time=round(time.time()-start, 2))

    except SQLAlchemyError as e:
        logger.error(f"DB error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Processing error")
