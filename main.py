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
import json
import re
from datetime import datetime

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

# --- Patch for GPT-5-mini unsupported "stop" ---
class PatchedChatOpenAI(ChatOpenAI):
    def _get_invocation_params(self, stop=None, **kwargs):
        params = super()._get_invocation_params(**kwargs)
        if "stop" in params:
            params.pop("stop")
        return params
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

# Prompt
SYSTEM_PROMPT = f"""
You are EnerBot, an expert Georgian electricity market data analyst...
{DB_SCHEMA_DOC}
"""

class Question(BaseModel):
    query: str

def convert_decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_decimal_to_float(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_decimal_to_float(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    return obj

# === Chart helpers ===
def is_chart_request(query: str) -> tuple[bool, str]:
    query_lower = query.lower()
    chart_keywords = ["chart","plot","graph","visualize","visualization","show as","display as","present as","draw","render as"]
    chart_patterns = ["bar chart","line chart","pie chart","show as bar","show as line","show as pie","display as chart","visualize as","plot as"]

    for pattern in chart_patterns:
        if pattern in query_lower:
            if "pie" in pattern: return True, "pie"
            if "line" in pattern: return True, "line"
            return True, "bar"

    is_chart = any(k in query_lower for k in chart_keywords)
    if not is_chart:
        return False, None
    return True, "auto"

def intelligent_chart_type_selection(raw_results, query: str, explicit_type: str = None):
    if explicit_type and explicit_type != "auto":
        return explicit_type
    if not raw_results or not isinstance(raw_results, list):
        return "bar"

    query_lower = query.lower()
    sample_row = raw_results[0]
    num_rows = len(raw_results)
    num_cols = len(sample_row) if isinstance(sample_row,(list,tuple)) else 0

    if any(w in query_lower for w in ["trend","over time","monthly","yearly","progression"]):
        return "line"
    if any(w in query_lower for w in ["share","proportion","percentage","distribution","composition","breakdown"]):
        return "pie"
    if any(w in query_lower for w in ["compare","vs","versus","between"]):
        return "bar"

    if num_cols >= 2:
        first_col = sample_row[0]
        if isinstance(first_col,str):
            for fmt in ['%Y-%m-%d','%Y-%m','%m/%d/%Y','%d/%m/%Y','%Y']:
                try:
                    datetime.strptime(first_col,fmt)
                    return "line"
                except: continue

    if num_rows <= 6: return "pie"
    if num_rows > 15: return "bar"
    return "bar"

def process_sql_results_for_chart(raw_results, query: str):
    if not raw_results: return [], {}
    chart_data, metadata = [], {"title":"Energy Data","xAxisTitle":"Category","yAxisTitle":"Value","datasetLabel":"Data"}
    query_lower = query.lower()

    for row in raw_results:
        row = convert_decimal_to_float(row)
        if not isinstance(row,(list,tuple)) or len(row)<2: continue

        if len(row)==2:
            col1,col2=row
            is_date=False
            if isinstance(col1,str):
                for fmt in ['%Y-%m-%d','%Y-%m','%m/%d/%Y','%d/%m/%Y']:
                    try:
                        datetime.strptime(col1,fmt)
                        is_date=True;break
                    except: continue
            if is_date:
                chart_data.append({"date":str(col1),"value":float(col2)})
                metadata["xAxisTitle"]="Date"
                metadata["datasetLabel"]="Value over Time"
            else:
                chart_data.append({"category":str(col1),"value":float(col2)})
        elif len(row)>=3:
            chart_data.append({"date":str(row[0]),"category":str(row[1]),"value":float(row[2])})
            metadata["xAxisTitle"]="Date"
            metadata["datasetLabel"]="By Category"
    return chart_data, metadata

def extract_json_from_text(text: str):
    if not text: return None
    matches = re.findall(r'\[[\s\S]*?\]', text)
    for m in matches:
        try:
            parsed = json.loads(m)
            if isinstance(parsed,list) and parsed: return convert_decimal_to_float(parsed)
        except: continue
    return None
# === End helpers ===

@app.get("/healthz")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(q: Question, x_app_key: str = Header(...)):
    if not APP_SECRET_KEY or x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        llm = PatchedChatOpenAI(model="gpt-5-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True, use_query_checker=True, top_k=50, return_intermediate_steps=True)

        is_chart, chart_type = is_chart_request(q.query)
        result = db_chain.invoke(SYSTEM_PROMPT + "\n\n" + q.query)
        raw_answer = result.get("result","")
        steps = result.get("intermediate_steps",[])
        chart_data, chart_metadata, final_answer = [], {}, raw_answer

        if is_chart:
            json_data = extract_json_from_text(raw_answer)
            if json_data:
                chart_data=json_data;final_answer="Here's your data visualization:";chart_metadata={"title":"Energy Data","xAxisTitle":"Category","yAxisTitle":"Value","datasetLabel":"Data"}
            else:
                for step in steps:
                    if isinstance(step,dict) and "sql_cmd" in step:
                        try:
                            with engine.connect() as conn:
                                rows = conn.execute(text(step["sql_cmd"])).fetchall()
                                if rows:
                                    chart_data, chart_metadata = process_sql_results_for_chart(rows,q.query)
                                    chart_type = intelligent_chart_type_selection(rows,q.query,chart_type)
                                    final_answer="Here's your data visualization:";break
                        except Exception as e: continue

        return {"answer": final_answer if not chart_data else "Here's your data visualization:","chartType": chart_type if chart_data else None,"data": chart_data if chart_data else None,"chartMetadata": chart_metadata if chart_data else None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
