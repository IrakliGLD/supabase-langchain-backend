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
- If unsure, return: "I don't know based on the data."

ANSWERING RULES
1. If no results:
   - Respond exactly: "I don't know based on the data."

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

def convert_decimal_to_float(obj):
    """Recursively convert Decimal objects to float"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_decimal_to_float(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_decimal_to_float(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    return obj

def is_chart_request(query: str) -> tuple[bool, str]:
    """Detect if user is asking for a chart and determine type"""
    query_lower = query.lower()
    
    # Chart request keywords
    chart_keywords = ["chart", "plot", "graph", "visualize", "show as", "display as"]
    is_chart = any(keyword in query_lower for keyword in chart_keywords)
    
    if not is_chart:
        return False, None
    
    # Determine chart type
    if any(word in query_lower for word in ["pie", "pie chart"]):
        return True, "pie"
    elif any(word in query_lower for word in ["line", "line chart", "trend"]):
        return True, "line"
    else:
        return True, "bar"  # default

def process_sql_results_for_chart(raw_results, query: str):
    """Process SQL results into chart-friendly format"""
    if not raw_results or not isinstance(raw_results, list):
        return []
    
    chart_data = []
    
    for row in raw_results:
        # Convert any Decimal objects to float
        row = convert_decimal_to_float(row)
        
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
            
        # Case 1: Two columns - could be (date, value) or (category, value)
        if len(row) == 2:
            col1, col2 = row
            
            # Try to detect if first column is a date
            is_date = False
            if isinstance(col1, str):
                try:
                    # Try parsing various date formats
                    for date_format in ['%Y-%m-%d', '%Y-%m', '%m/%d/%Y', '%d/%m/%Y']:
                        try:
                            datetime.strptime(col1, date_format)
                            is_date = True
                            break
                        except ValueError:
                            continue
                except:
                    pass
            
            if is_date:
                chart_data.append({"date": str(col1), "value": float(col2)})
            else:
                # Categorical data
                chart_data.append({"sector": str(col1), "volume_tj": float(col2)})
        
        # Case 2: Three columns - often (period/year, category, value)
        elif len(row) == 3:
            col1, col2, col3 = row
            
            # Check if first column looks like a date/period
            if isinstance(col1, str) and ('-' in col1 or '/' in col1):
                chart_data.append({"date": str(col1), "sector": str(col2), "value": float(col3)})
            else:
                # Treat as (year, sector, value) or similar
                chart_data.append({"sector": str(col2), "volume_tj": float(col3)})
        
        # Case 3: Four or more columns - assume (period, sector, source, value)
        elif len(row) >= 4:
            period, sector, source, value = row[:4]
            chart_data.append({
                "date": str(period),
                "sector": str(sector), 
                "energy_source": str(source),
                "volume_tj": float(value)
            })
    
    return chart_data

def extract_json_from_text(text: str):
    """Extract JSON array from text response if present"""
    if not text:
        return None
        
    # Look for JSON array patterns
    json_pattern = r'\[[\s\S]*?\]'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list) and len(parsed) > 0:
                return convert_decimal_to_float(parsed)
        except json.JSONDecodeError:
            continue
    
    return None

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

        # Check if this is a chart request
        is_chart, chart_type = is_chart_request(q.query)
        
        result = db_chain.invoke(SYSTEM_PROMPT + "\n\n" + q.query)
        
        # Get the raw result
        raw_answer = result.get("result", "")
        intermediate_steps = result.get("intermediate_steps", [])
        
        print(f"Raw answer: {raw_answer}")
        print(f"Intermediate steps: {intermediate_steps}")
        
        chart_data = []
        final_answer = raw_answer
        
        if is_chart:
            # First, try to extract JSON from the text answer
            json_data = extract_json_from_text(str(raw_answer))
            if json_data:
                chart_data = json_data
                final_answer = "Here's your data visualization:"
            else:
                # If no JSON in answer, try to get raw SQL results
                if intermediate_steps:
                    for step in intermediate_steps:
                        if isinstance(step, dict) and 'sql_cmd' in step:
                            # Try to execute the SQL query directly to get raw results
                            try:
                                sql_query = step['sql_cmd']
                                with engine.connect() as conn:
                                    result_proxy = conn.execute(text(sql_query))
                                    raw_results = result_proxy.fetchall()
                                    print(f"Raw SQL results: {raw_results}")
                                    
                                    if raw_results:
                                        chart_data = process_sql_results_for_chart(raw_results, q.query)
                                        final_answer = "Here's your data visualization:"
                                        break
                            except Exception as e:
                                print(f"Error executing SQL directly: {e}")
                                continue
        
        # Clean up the response
        response = {
            "answer": final_answer if not chart_data else "Here's your data visualization:",
            "chartType": chart_type if chart_data else None,
            "data": chart_data if chart_data else None
        }
        
        print(f"Final response: {response}")
        return response

    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
