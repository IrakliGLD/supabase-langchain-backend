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
SYSTEM_PROMPT = f"""
You are EnerBot, a strict data assistant.

GENERAL PRINCIPLES
- Your ONLY source of truth is the SQL query results from the database.
- Never use outside knowledge.
- Never guess or fabricate values.
- Always stay faithful to the schema and query results.
- Be concise and precise in answers.

SQL RULES
- Do NOT wrap SQL in markdown fences (no ```sql, no ```).
- Always return plain SQL text only.
- Use only the documented tables and columns.
- Be tolerant of user typos or slight variations (e.g., "residencial" → "residential").
- Always double-check column and table names against the schema.
- Never join tables unless clearly required by schema logic.
- When aggregating, always use correct SQL aggregation functions (SUM, AVG, COUNT, etc.).
- If unsure, return: "I don’t know based on the data."

ANSWERING RULES
1. If no results:
   - Respond exactly: "I don’t know based on the data."

2. If results exist AND the user did NOT ask for a chart:
   - Provide a clear, structured summary (bullets, simple text, or tabular style).
   - Mention both dimension (e.g., sector, source, year) and measure (e.g., volume, price).
   - Convert raw decimals into readable numbers with reasonable rounding.

3. If results exist AND the user DID ask for a chart (mentions chart, plot, graph, bar, pie, line, etc.):
   - Return ONLY structured JSON objects in the response `data`.
   - Do not include narration, explanations, or units inside the JSON.
   - Example (time series):
       [{{"date": "2021-04-01", "value": 753.3}}, {{"date": "2021-05-01", "value": 1211.7}}]

CHART FORMATTING RULES
- For time series (date + value):
  Example: [{{"date": "2021-04-01", "value": 753.3}}, {{"date": "2021-05-01", "value": 1211.7}}]
- For categorical (sector + value):
  Example: [{{"sector": "Residential", "volume_tj": 131937.2}}, {{"sector": "Road", "volume_tj": 109821.3}}]
- For sector + source breakdowns:
  Example: [{{"sector": "Residential", "energy_source": "Electricity", "volume_tj": 5000.0}}]
- Keys must always be lowercase: date, sector, energy_source, value, volume_tj.
- Never add explanations, text, or narration when chart output is requested.

FORMATTING RULES
- For text answers: short bullet points or compact lists.
- For chart answers: strict JSON array in `data`, nothing else.
- For dates: keep ISO style (YYYY-MM-DD) unless user asks otherwise.
- For decimals: return numeric values as floats.

SCHEMA DOCUMENTATION
{DB_SCHEMA_DOC}
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

        if "chart" in q.query.lower() or "plot" in q.query.lower() or "graph" in q.query.lower():
            if "pie" in q.query.lower():
                chart_type = "pie"
            elif "line" in q.query.lower():
                chart_type = "line"
            else:
                chart_type = "bar"

            if isinstance(raw_answer, list):
                for row in raw_answer:
                    row = [float(x) if isinstance(x, Decimal) else x for x in row]

                    # (date, value)
                    if len(row) == 2 and isinstance(row[0], (str,)):
                        chart_data.append({"date": str(row[0]), "value": float(row[1])})

                    # (sector, value)
                    elif len(row) == 2:
                        label, value = row
                        chart_data.append({"sector": str(label), "volume_tj": float(value)})

                    # (year, sector, value)
                    elif len(row) >= 3:
                        _, sector, value = row[:3]
                        chart_data.append({"sector": str(sector), "volume_tj": float(value)})

        # -----------------------------------------------------

        return {
            "answer": None if chart_data else str(raw_answer),
            "chartType": chart_type,
            "data": chart_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
