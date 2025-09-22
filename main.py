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

# Improved system prompt
SYSTEM_PROMPT = f"""
You are EnerBot, an expert Georgian electricity market data analyst with strict data integrity principles.

=== CORE PRINCIPLES ===
🔒 DATA INTEGRITY: Your ONLY source of truth is the SQL query results from the database. Never use outside knowledge, assumptions, or estimates.
📊 ACCURACY FIRST: Be precise with numbers, dates, and categories. Round decimals appropriately but maintain data fidelity.
🎯 RELEVANT RESULTS: Focus on what the user actually asked for. Don't provide tangential information.
🚫 NO HALLUCINATION: If unsure about anything, respond: "I don't know based on the available data."

=== SQL QUERY RULES ===
✅ CLEAN SQL ONLY: Return plain SQL text without markdown fences (no ```sql, no ```).
✅ SCHEMA COMPLIANCE: Use only documented tables/columns. Double-check all names against the schema.
✅ FLEXIBLE MATCHING: Handle user typos gracefully (e.g., "residencial" → "residential", "elektric" → "electricity").
✅ PROPER AGGREGATION: Use correct SQL functions (SUM for totals, AVG for averages, COUNT for quantities).
✅ SMART FILTERING: Apply appropriate WHERE clauses for date ranges, sectors, and energy sources.
✅ LOGICAL JOINS: Only join tables when schema relationships clearly support it.
✅ PERFORMANCE AWARE: Use LIMIT clauses for large datasets, especially for charts.

=== RESPONSE FORMATTING ===

📝 FOR TEXT ANSWERS (when user does NOT request charts):
- Provide clear, structured summaries
- Use bullet points or tables for multiple data points
- Include context: time periods, sectors, units of measurement
- Round numbers appropriately (e.g., "1,083.9 TJ" not "1083.87439 TJ")
- Highlight key insights or trends when relevant

📈 FOR CHART REQUESTS (when user mentions: chart, plot, graph, visualize, show as):
- Let the backend handle data formatting - just return natural language summary
- Mention what time period, sectors, or categories are included
- Note any significant patterns or outliers
- Keep explanations brief since the chart will show the details

=== QUERY OPTIMIZATION ===
🔍 TIME SERIES: For monthly/yearly trends, ensure proper date ordering (ORDER BY date/period)
🔍 CATEGORIZATION: Group by relevant dimensions (sector, energy_source, entity)
🔍 AGGREGATION: Sum volumes, average prices, count occurrences as appropriate
🔍 FILTERING: Apply reasonable date ranges if not specified (e.g., last 12 months)

=== COMMON PATTERNS ===
• Energy consumption by sector → GROUP BY sector, SUM(volume_tj)
• Monthly trends → GROUP BY date/period, ORDER BY date
• Price comparisons → SELECT entity, price, date for relevant periods
• Market share → Calculate percentages using window functions
• Import/export data → Use trade table with appropriate entity filters

=== ERROR HANDLING ===
❌ No data found → "I don't have data for that specific request."
❌ Ambiguous request → Ask for clarification: "Could you specify the time period/sector?"
❌ Invalid parameters → Suggest alternatives based on available data

=== SCHEMA DOCUMENTATION ===
{DB_SCHEMA_DOC}

=== EXAMPLES ===
Good: "Residential sector consumed 131,937.2 TJ in 2022, representing 45% of total energy use."
Bad: "Energy consumption is typically high in residential areas due to heating and cooling needs."

Good: "Natural gas prices ranged from $2.15 to $4.78 per unit between Jan-Dec 2022."
Bad: "Natural gas prices have been volatile recently due to global market conditions."

REMEMBER: You are a data analyst, not a general energy expert. Stick to what the data shows!
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
    """Process SQL results into chart-friendly format with metadata"""
    if not raw_results or not isinstance(raw_results, list):
        return [], {}
    
    chart_data = []
    metadata = {
        "title": "Energy Data",
        "xAxisTitle": "Category",
        "yAxisTitle": "Value",
        "datasetLabel": "Data"
    }
    
    # Analyze query to determine appropriate labels
    query_lower = query.lower()
    
    # Determine Y-axis title based on common patterns
    if any(word in query_lower for word in ["volume", "consumption", "energy"]):
        if "tj" in query_lower or "terajoule" in query_lower:
            metadata["yAxisTitle"] = "Volume (TJ)"
        else:
            metadata["yAxisTitle"] = "Energy Volume"
    elif any(word in query_lower for word in ["price", "cost", "tariff"]):
        metadata["yAxisTitle"] = "Price"
    elif any(word in query_lower for word in ["count", "number", "quantity"]):
        metadata["yAxisTitle"] = "Count"
    elif any(word in query_lower for word in ["percentage", "percent", "share"]):
        metadata["yAxisTitle"] = "Percentage (%)"
    
    # Determine chart title based on query content
    if "residential" in query_lower:
        metadata["title"] = "Residential Energy Data"
    elif "commercial" in query_lower:
        metadata["title"] = "Commercial Energy Data"
    elif "industrial" in query_lower:
        metadata["title"] = "Industrial Energy Data"
    elif "monthly" in query_lower or "month" in query_lower:
        metadata["title"] = "Monthly Energy Trends"
    elif "yearly" in query_lower or "year" in query_lower or "annual" in query_lower:
        metadata["title"] = "Annual Energy Data"
    elif "sector" in query_lower:
        metadata["title"] = "Energy Data by Sector"
    elif "source" in query_lower:
        metadata["title"] = "Energy Data by Source"
    elif "price" in query_lower:
        metadata["title"] = "Energy Prices"
    elif "trade" in query_lower or "import" in query_lower or "export" in query_lower:
        metadata["title"] = "Energy Trade Data"
    
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
                metadata["xAxisTitle"] = "Date"
                metadata["datasetLabel"] = "Value over Time"
                if not metadata.get("title_set"):
                    metadata["title"] = "Time Series Analysis"
            else:
                # Categorical data
                chart_data.append({"sector": str(col1), "volume_tj": float(col2)})
                metadata["xAxisTitle"] = "Sector" if "sector" in query_lower else "Category"
                metadata["datasetLabel"] = "Energy Volume"
        
        # Case 2: Three columns - often (period/year, category, value)
        elif len(row) == 3:
            col1, col2, col3 = row
            
            # Check if first column looks like a date/period
            if isinstance(col1, str) and ('-' in col1 or '/' in col1):
                chart_data.append({"date": str(col1), "sector": str(col2), "value": float(col3)})
                metadata["xAxisTitle"] = "Date"
                metadata["datasetLabel"] = str(col2)
                if not metadata.get("title_set"):
                    metadata["title"] = "Time Series by Category"
            else:
                # Treat as (year, sector, value) or similar
                chart_data.append({"sector": str(col2), "volume_tj": float(col3)})
                metadata["xAxisTitle"] = "Sector"
                metadata["datasetLabel"] = "Energy Volume"
        
        # Case 3: Four or more columns - assume (period, sector, source, value)
        elif len(row) >= 4:
            period, sector, source, value = row[:4]
            chart_data.append({
                "date": str(period),
                "sector": str(sector), 
                "energy_source": str(source),
                "volume_tj": float(value)
            })
            metadata["xAxisTitle"] = "Date"
            metadata["datasetLabel"] = "Energy Sources"
            if not metadata.get("title_set"):
                metadata["title"] = "Energy Sources Over Time"
    
    return chart_data, metadata

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
            chart_metadata = {}
            
            if json_data:
                chart_data = json_data
                # Create basic metadata for JSON data
                chart_metadata = {
                    "title": "Energy Data Visualization",
                    "xAxisTitle": "Category",
                    "yAxisTitle": "Value",
                    "datasetLabel": "Data"
                }
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
                                        chart_data, chart_metadata = process_sql_results_for_chart(raw_results, q.query)
                                        final_answer = "Here's your data visualization:"
                                        break
                            except Exception as e:
                                print(f"Error executing SQL directly: {e}")
                                continue
        
        # Clean up the response
        response = {
            "answer": final_answer if not chart_data else "Here's your data visualization:",
            "chartType": chart_type if chart_data else None,
            "data": chart_data if chart_data else None,
            "chartMetadata": chart_metadata if chart_data else None
        }
        
        print(f"Final response: {response}")
        return response

    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
