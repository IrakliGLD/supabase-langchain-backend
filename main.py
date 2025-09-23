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

# Enhanced system prompt with read-only access and clarification guidance
SYSTEM_PROMPT = f"""
You are EnerBot, an expert Georgian electricity market data analyst.

=== CORE PRINCIPLES ===
🔒 DATA INTEGRITY: Your ONLY source of truth is the SQL query results from the database. Never use outside knowledge.
🔒 READ-ONLY ACCESS: You can ONLY SELECT/READ data from the database. You CANNOT modify any data.
📊 CHART REQUESTS: When users ask for charts, return clean tabular data that can be processed into visualizations.

=== CRITICAL: CHART DATA FORMATTING ===
When users request charts (mentions: chart, plot, graph, visualize, show as, display as):
- Return SQL results in a clean, structured format
- For time series queries: return (date, value) pairs
- For categorical queries: return (category, value) pairs  
- Keep explanations minimal - focus on returning data

=== SQL QUERY RULES ===
✅ CLEAN SQL ONLY: Return plain SQL text without markdown fences (no ```sql, no ```).
✅ SCHEMA COMPLIANCE: Use only documented tables/columns.
✅ PROPER AGGREGATION: Use correct SQL functions (SUM, AVG, COUNT).
✅ DATE ORDERING: Always ORDER BY date for time series data.

=== CLARIFICATION GUIDELINES ===
When queries are ambiguous, ask specific clarifying questions:
- Price queries: "Do you want prices in GEL/MWh or USD/MWh?"
- Vague time periods: "Which specific time period? (months/years)"
- Multiple data types: "Which type of data? (generation, consumption, prices)"

=== SCHEMA DOCUMENTATION ===
{DB_SCHEMA_DOC}

REMEMBER: For chart requests, focus on returning clean, parseable data rather than detailed explanations.
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
    
    # Chart request keywords - comprehensive list
    chart_patterns = [
        "chart", "plot", "graph", "visualize", "visualization", 
        "show as", "display as", "present as", "render as",
        "bar chart", "line chart", "pie chart"
    ]
    
    # Check if any chart pattern is present
    is_chart = any(pattern in query_lower for pattern in chart_patterns)
    
    print(f"🔍 Chart Detection:")
    print(f"   Query: '{query}'")
    print(f"   Is Chart: {is_chart}")
    
    if not is_chart:
        return False, None
    
    # Determine specific chart type
    if any(word in query_lower for word in ["pie", "pie chart"]):
        chart_type = "pie"
    elif any(word in query_lower for word in ["line", "line chart", "trend", "over time"]):
        chart_type = "line"
    else:
        chart_type = "bar"  # default
    
    print(f"   Chart Type: {chart_type}")
    return True, chart_type

def intelligent_chart_type_selection(raw_results, query: str, explicit_type: str = None):
    """Intelligently select the best chart type based on data characteristics"""
    
    if explicit_type and explicit_type != "auto":
        return explicit_type
    
    if not raw_results or len(raw_results) == 0:
        return "bar"
    
    query_lower = query.lower()
    num_rows = len(raw_results)
    sample_row = raw_results[0] if raw_results else []
    
    print(f"🧠 Chart Type Selection:")
    print(f"   Data rows: {num_rows}")
    print(f"   Sample row: {sample_row}")
    
    # Rule 1: Query context clues
    if any(word in query_lower for word in ["trend", "over time", "monthly", "yearly"]):
        print(f"   Decision: LINE (temporal context)")
        return "line"
    
    if any(word in query_lower for word in ["share", "proportion", "breakdown"]):
        print(f"   Decision: PIE (composition context)")
        return "pie"
    
    # Rule 2: Data structure analysis
    if len(sample_row) >= 2:
        first_col = sample_row[0]
        # Check if first column is date-like
        if isinstance(first_col, str) and ('-' in first_col or '/' in first_col):
            try:
                datetime.strptime(str(first_col)[:10], '%Y-%m-%d')
                print(f"   Decision: LINE (date structure detected)")
                return "line"
            except:
                pass
    
    # Rule 3: Default based on number of items
    if num_rows <= 6:
        print(f"   Decision: PIE (few categories)")
        return "pie"
    
    print(f"   Decision: BAR (default)")
    return "bar"

def process_sql_results_for_chart(raw_results, query: str):
    """Process SQL results into chart-friendly format with comprehensive debugging"""
    
    print(f"\n📊 Processing SQL Results for Chart:")
    print(f"   Raw results count: {len(raw_results) if raw_results else 0}")
    print(f"   Query: '{query}'")
    
    if not raw_results:
        print(f"   ❌ No raw results to process")
        return [], {}
    
    # Print first few results for debugging
    for i, row in enumerate(raw_results[:3]):
        print(f"   Row {i}: {row}")
    
    chart_data = []
    metadata = {
        "title": "Energy Data Visualization",
        "xAxisTitle": "Category",
        "yAxisTitle": "Value",
        "datasetLabel": "Data"
    }
    
    # Analyze query for better metadata
    query_lower = query.lower()
    
    if "hydro" in query_lower and "generation" in query_lower:
        metadata["title"] = "Hydro Power Generation"
        metadata["yAxisTitle"] = "Generation (MWh)"
        metadata["datasetLabel"] = "Hydro Generation"
    
    if any(word in query_lower for word in ["april", "may", "june", "monthly"]):
        metadata["xAxisTitle"] = "Month"
    
    # Process each row
    for i, row in enumerate(raw_results):
        print(f"   Processing row {i}: {row}")
        
        # Convert decimals to float
        row = convert_decimal_to_float(row)
        
        if len(row) >= 2:
            col1, col2 = row[0], row[1]
            
            # Try to detect date in first column
            is_date = False
            if isinstance(col1, str):
                # Check various date formats
                try:
                    # Try YYYY-MM-DD format
                    datetime.strptime(col1, '%Y-%m-%d')
                    is_date = True
                except:
                    try:
                        # Try YYYY-MM format
                        datetime.strptime(col1, '%Y-%m')
                        is_date = True
                    except:
                        pass
            
            if is_date:
                value = float(col2) if col2 is not None else 0.0
                chart_item = {"date": str(col1), "value": value}
                chart_data.append(chart_item)
                print(f"     → Time series: {chart_item}")
                metadata["xAxisTitle"] = "Date"
            else:
                value = float(col2) if col2 is not None else 0.0
                chart_item = {"sector": str(col1), "volume_tj": value}
                chart_data.append(chart_item)
                print(f"     → Category: {chart_item}")
                metadata["xAxisTitle"] = "Category"
    
    print(f"   ✅ Final chart data: {len(chart_data)} items")
    print(f"   📋 Metadata: {metadata}")
    
    return chart_data, metadata

def create_test_response(query: str):
    """Create test response for debugging"""
    print(f"\n🧪 Creating test response for: '{query}'")
    
    # Mock hydro generation data for testing
    if "hydro" in query.lower():
        test_data = [
            {"date": "2022-04-01", "value": 1083.9},
            {"date": "2022-05-01", "value": 1452.7},
            {"date": "2022-06-01", "value": 1475.4},
            {"date": "2022-07-01", "value": 1297.1},
            {"date": "2022-08-01", "value": 1153.1},
        ]
        
        metadata = {
            "title": "Hydro Power Generation",
            "xAxisTitle": "Month",
            "yAxisTitle": "Generation (MWh)", 
            "datasetLabel": "Monthly Generation"
        }
        
        return test_data, metadata, "bar"
    
    return [], {}, None

@app.get("/healthz")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-chart")
def test_chart(q: Question):
    """Test endpoint for debugging chart generation"""
    print(f"\n🔧 TEST CHART ENDPOINT")
    print(f"Query: {q.query}")
    
    is_chart, chart_type = is_chart_request(q.query)
    
    if not is_chart:
        return {
            "answer": "This doesn't appear to be a chart request",
            "chartType": None,
            "data": None,
            "chartMetadata": None
        }
    
    # Create test data
    chart_data, chart_metadata, chart_type = create_test_response(q.query)
    
    response = {
        "answer": "Here's your test data visualization:",
        "chartType": chart_type,
        "data": chart_data,
        "chartMetadata": chart_metadata
    }
    
    print(f"✅ Test response: {response}")
    return response

@app.post("/ask")
def ask(q: Question, x_app_key: str = Header(...)):
    # 🔒 API Key Authentication
    if not APP_SECRET_KEY or x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        print(f"\n🚀 Processing Query: '{q.query}'")
        
        # Check if this is a chart request
        is_chart, chart_type = is_chart_request(q.query)
        
        if is_chart:
            print(f"📊 Chart request detected - Type: {chart_type}")
            
            # For debugging, let's try direct SQL execution first
            if "hydro" in q.query.lower() and "generation" in q.query.lower():
                print(f"🔍 Direct SQL test for hydro generation")
                
                # Try a simple direct query for testing
                try:
                    test_sql = """
                    SELECT 
                        DATE_TRUNC('month', period)::date as month,
                        SUM(volume_tj) as generation_mwh
                    FROM energy_balance_long 
                    WHERE entity_name ILIKE '%hydro%' 
                        AND period >= '2022-04-01' 
                        AND period <= '2022-12-31'
                    GROUP BY DATE_TRUNC('month', period)
                    ORDER BY month
                    """
                    
                    print(f"🔧 Test SQL: {test_sql}")
                    
                    with engine.connect() as conn:
                        result = conn.execute(text(test_sql))
                        raw_results = result.fetchall()
                        print(f"📋 Direct SQL results: {raw_results}")
                        
                        if raw_results:
                            chart_data, chart_metadata = process_sql_results_for_chart(raw_results, q.query)
                            optimal_chart_type = intelligent_chart_type_selection(raw_results, q.query, chart_type)
                            
                            response = {
                                "answer": "Here's your hydro generation data:",
                                "chartType": optimal_chart_type,
                                "data": chart_data,
                                "chartMetadata": chart_metadata
                            }
                            
                            print(f"✅ Direct SQL response: {response}")
                            return response
                            
                except Exception as e:
                    print(f"❌ Direct SQL failed: {e}")
        
        # Fallback to normal LangChain processing
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

        print(f"🤖 Using LangChain for query processing")
        result = db_chain.invoke(SYSTEM_PROMPT + "\n\n" + q.query)
        
        raw_answer = result.get("result", "")
        intermediate_steps = result.get("intermediate_steps", [])
        
        print(f"🔍 LangChain result: {raw_answer}")
        print(f"🔍 Intermediate steps count: {len(intermediate_steps)}")
        
        chart_data = []
        chart_metadata = {}
        final_answer = raw_answer
        
        # Process chart data if this was a chart request
        if is_chart and intermediate_steps:
            print(f"📊 Processing LangChain results for chart")
            
            for i, step in enumerate(intermediate_steps):
                print(f"   Step {i}: {type(step)} - {step}")
                
                if isinstance(step, dict):
                    sql_query = step.get('sql_cmd') or step.get('query') or step.get('sql')
                    
                    if sql_query:
                        try:
                            print(f"🔧 Executing SQL: {sql_query}")
                            with engine.connect() as conn:
                                result_proxy = conn.execute(text(sql_query))
                                raw_results = result_proxy.fetchall()
                                
                                if raw_results:
                                    chart_data, chart_metadata = process_sql_results_for_chart(raw_results, q.query)
                                    chart_type = intelligent_chart_type_selection(raw_results, q.query, chart_type)
                                    final_answer = "Here's your data visualization:"
                                    break
                        except Exception as e:
                            print(f"❌ SQL execution error: {e}")
                            continue

        response = {
            "answer": final_answer,
            "chartType": chart_type if chart_data else None,
            "data": chart_data if chart_data else None,
            "chartMetadata": chart_metadata if chart_data else None
        }
        
        print(f"✅ Final response: {response}")
        return response

    except Exception as e:
        print(f"❌ Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
