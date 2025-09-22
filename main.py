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
    """Remove markdown fences if GPT accidentally adds them."""
    return query.replace("```sql", "").replace("```", "").strip()

old_execute = db._execute
def cleaned_execute(sql: str, *args, **kwargs):
    sql = clean_sql(sql)
    return old_execute(sql, *args, **kwargs)
db._execute = cleaned_execute
# ----------------------------------------------------------

# FastAPI app
app = FastAPI(title="Supabase LangChain Backend")

# CORS settings (open for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated system prompt
SYSTEM_PROMPT = f"""You are a data assistant.
You must ONLY answer using the SQL query results from the database.

⚠️ Important rules for SQL:
- Do NOT wrap SQL in markdown fences (no ```sql, no ```).
- Return plain SQL text only.
- If a user query contains a possible typo (e.g., "residencial" instead of "residential"),
  infer the closest valid column name or sector value from the schema and use that instead.

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
    # 🔒 API Key Authentication
    if not APP_SECRET_KEY or x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Always use server's own OpenAI key
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
            return_intermediate_steps=True
        )

        result = db_chain.invoke(SYSTEM_PROMPT + "\n\n" + q.query)
        raw_answer = result.get("result", result)

        # ---------------- Normalize response ----------------
        chart_type = None
        chart_data = []

        if "chart" in q.query.lower():
            chart_type = "pie" if "pie" in q.query.lower() else "bar"

            if isinstance(raw_answer, list):
                for row in raw_answer:
                    # Convert decimals & tuples into dicts
                    row = [float(x) if isinstance(x, Decimal) else x for x in row]

                    # Case 1: ('Residential', 131937.2)
                    if len(row) == 2:
                        label, value = row
                        chart_data.append({"label": str(label), "value": float(value)})

                    # Case 2: (2022, 'Residential', 131937.2)
                    elif len(row) >= 3:
                        _, sector, value = row[:3]
                        chart_data.append({"sector": str(sector), "volume_tj": float(value)})
        # -----------------------------------------------------

        return {
            "answer": str(raw_answer),
            "chartType": chart_type,
            "data": chart_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
