import os
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from dotenv import load_dotenv
from decimal import Decimal

# Import DB documentation context
from context import DB_SCHEMA_DOC

# Load .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

if not OPENAI_API_KEY or not SUPABASE_DB_URL:
    raise RuntimeError("Missing OPENAI_API_KEY or SUPABASE_DB_URL in environment")

# Supabase (Postgres) DB connection
engine = create_engine(SUPABASE_DB_URL, pool_pre_ping=True)

# Restrict DB access ONLY to documented tables/views
allowed_tables = [
    "energy_balance_long",
    "entities",
    "monthly_cpi",
    "price",
    "tariff_gen",
    "tech_quantity",
    "trade"
]
db = SQLDatabase(engine, include_tables=allowed_tables)

# --- Patch execution to clean accidental markdown fences ---
def clean_sql(query: str) -> str:
    return query.replace("```sql", "").replace("```", "").strip()

old_execute = db._execute
def cleaned_execute(sql: str, *args, **kwargs):
    sql = clean_sql(sql)
    return old_execute(sql, *args, **kwargs)
db._execute = cleaned_execute
# ----------------------------------------------------------

# Normalize SQL result tuples -> dicts
def normalize_result(result, keys=None):
    if not result:
        return []
    normalized = []
    for row in result:
        if isinstance(row, tuple) and keys:
            normalized.append({k: (float(v) if isinstance(v, Decimal) else v) for k, v in zip(keys, row)})
        elif isinstance(row, dict):
            normalized.append({k: (float(v) if isinstance(v, Decimal) else v) for k, v in row.items()})
    return normalized

# FastAPI app
app = FastAPI(title="Supabase LangChain Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = f"""You are a data assistant.
You must ONLY answer using the SQL query results from the database.

⚠️ Important rules for SQL:
- Do NOT wrap SQL in markdown fences (no ```sql, no ```).
- Return plain SQL text only.
- If a user query contains a possible typo (e.g., "residencial" instead of "residential"),
  infer the closest valid column name or sector value from the schema and use that instead.

📊 Chart instructions:
- If the user asks for a "pie chart", return tabular data + set chartType="pie".
- If the user asks for a "bar chart" or "histogram", return tabular data + set chartType="bar".
- If the user asks for a "line chart", return tabular data + set chartType="line".
- If no chart is requested, chartType=null.

Use the following schema documentation for context:
{DB_SCHEMA_DOC}

If results are empty or insufficient, reply exactly: "I don’t know based on the data."
Never use outside knowledge.
"""

class Question(BaseModel):
    query: str

@app.get("/healthz")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/introspect")
def introspect():
    try:
        schema = db.get_table_info()
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(q: Question, x_app_key: str = Header(...)):
    if not APP_SECRET_KEY or x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )

        db_chain = SQLDatabaseChain.from_llm(
            llm=llm,
            db=db,
            verbose=True,
            use_query_checker=True,
            top_k=50,
            return_direct=True
        )

        raw_result = db_chain.run(SYSTEM_PROMPT + "\n\n" + q.query)

        response = {
            "answer": None,
            "chartData": None,
            "chartType": None
        }

        # Detect chart type from query
        query_lower = q.query.lower()
        if "pie chart" in query_lower:
            response["chartType"] = "pie"
        elif "bar chart" in query_lower or "histogram" in query_lower:
            response["chartType"] = "bar"
        elif "line chart" in query_lower:
            response["chartType"] = "line"

        if isinstance(raw_result, list):
            if raw_result and isinstance(raw_result[0], tuple):
                keys = ["year", "sector", "value"] if len(raw_result[0]) == 3 else None
                normalized = normalize_result(raw_result, keys)
                response["answer"] = "Here’s the chart:"
                response["chartData"] = normalized
            elif raw_result and isinstance(raw_result[0], dict):
                normalized = normalize_result(raw_result)
                response["answer"] = "Here’s the chart:"
                response["chartData"] = normalized
            else:
                response["answer"] = str(raw_result)
        else:
            response["answer"] = str(raw_result)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
