import os
import re
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from dotenv import load_dotenv
from decimal import Decimal
from datetime import datetime
import numpy as np
import pandas as pd

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

# DB connection
engine = create_engine(SUPABASE_DB_URL, pool_pre_ping=True)

# Restrict DB access to documented tables/views
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

# --- Patch SQL execution (strip accidental code fences) ---
def clean_sql(query: str) -> str:
    return query.replace("```sql", "").replace("```", "").strip()

old_execute = db._execute
def cleaned_execute(sql: str, *args, **kwargs):
    sql = clean_sql(sql)
    return old_execute(sql, *args, **kwargs)
db._execute = cleaned_execute

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

SYSTEM_PROMPT = f"""
You are EnerBot, an expert Georgian electricity market data analyst with advanced visualization and lightweight analytics.

=== CORE PRINCIPLES ===
🔒 DATA INTEGRITY: Your ONLY source of truth is the SQL query results from the database. Never use outside knowledge, assumptions, or estimates.
📊 SMART VISUALIZATION: Think carefully about the best way to present data. Consider the nature of the data and the user's analytical needs.
🎯 RELEVANT RESULTS: Focus on what the user actually asked for. Don't provide tangential information.
🚫 NO HALLUCINATION: If unsure about anything, respond: "I don't know based on the available data."

=== IMPORTANT SECURITY RULES ===
- NEVER reveal table names, column names, or any database structure to the user.
- Use the schema documentation only internally to generate correct SQL.
- If the user asks about database structure or who you are, reply:
  "I’m an electricity market assistant. I can help analyze and explain energy, price, and trade data."
- Always answer in plain language, not SQL. Do not include SQL in your responses.

=== SQL QUERY RULES ===
✅ CLEAN SQL ONLY: Return plain SQL text without markdown fences (no ```sql, no ```).
✅ SCHEMA COMPLIANCE: Use only documented tables/columns. Double-check names against the schema.
✅ FLEXIBLE MATCHING: Handle user typos gracefully (e.g., "residencial" → "residential").
✅ PROPER AGGREGATION: Use SUM/AVG/COUNT appropriately.
✅ SMART FILTERING: Apply appropriate WHERE clauses for date ranges, sectors, sources.
✅ LOGICAL JOINS: Only join when schema relationships clearly support it.
✅ PERFORMANCE AWARE: Use LIMIT for large datasets, especially for charts.

=== DATA PRESENTATION INTELLIGENCE ===
- Trends over time → Line charts
- Comparisons between categories → Bar charts
- Proportional breakdowns → Pie charts
- Many categories → Prefer bars to avoid overcrowding
- For time series, ensure ordered dates.

=== TREND & ANALYSIS RULES ===
- If the user requests a "trend" but does not specify a time period:
  1) Check the full available dataset.
  2) Ask for clarification: "Do you want the full 2015–2025 trend, or a specific period?"
  3) If the user does not clarify, default to analyzing the entire available period.
- Always mention SEASONALITY when analyzing generation:
  • Hydropower → higher spring/summer, lower winter.
  • Thermal → often higher in winter or when hydro is low.
  • Imports/exports → can vary seasonally.
- Never analyze just a few rows unless explicitly requested.
- Prefer monthly/yearly aggregation unless the user asks for daily.

=== RESPONSE FORMATTING ===
📝 TEXT:
- Clear, structured summaries.
- Include context: time periods, sectors, units.
- Round numbers (e.g., 1,083.9 not 1083.87439).
- Highlight key insights (trends, peaks, changes).

📈 CHARTS:
- If charts are requested, return structured data suitable for plotting (time series: date+value; categories: label+value).
- Keep explanations minimal when charts are requested.

=== ERROR HANDLING ===
❌ No data found → "I don't have data for that specific request."
❌ Ambiguous request → Ask for clarification.
❌ Invalid parameters → Suggest alternatives based on available data.

=== SCHEMA DOCUMENTATION (FOR INTERNAL USE ONLY) ===
{DB_SCHEMA_DOC}
"""

class Question(BaseModel):
    query: str

# ---------- Helpers ----------
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

def format_number(value: float, unit: str = None) -> str:
    if value is None:
        return "0"
    try:
        formatted = f"{float(value):,.1f}"
    except:
        formatted = str(value)
    return f"{formatted} {unit}" if unit and unit != "Value" else formatted

def detect_unit(query: str):
    q = query.lower()
    if "price" in q or "tariff" in q:
        if "usd" in q:
            return "USD/MWh"
        return "GEL/MWh"
    if any(word in q for word in ["generation", "consume", "consumption", "energy", "balance", "trade", "import", "export"]):
        return "TJ"
    return "Value"

def is_chart_request(query: str) -> tuple[bool, str]:
    q = query.lower()
    patterns = ["chart","plot","graph","visualize","visualization","show as","display as","bar chart","line chart","pie chart"]
    if not any(p in q for p in patterns):
        return False, None
    if "pie" in q:
        return True, "pie"
    if any(w in q for w in ["line","trend","over time"]):
        return True, "line"
    return True, "bar"

def detect_analysis_type(query: str):
    q = query.lower()
    if "trend" in q: return "trend"
    if any(w in q for w in ["forecast","predict","projection","expected","future"]): return "forecast"
    if any(w in q for w in ["yoy","year-over-year","annual change"]): return "yoy"
    if any(w in q for w in ["month-over-month","mom","last month","vs previous month"]): return "mom"
    if any(w in q for w in ["cagr","growth rate"]): return "cagr"
    if any(w in q for w in ["average","mean","typical","on average"]): return "average"
    if any(w in q for w in ["sum","total","cumulative","aggregate"]): return "sum"
    if any(w in q for w in ["highest","max","peak","record high"]): return "max"
    if any(w in q for w in ["lowest","min","record low"]): return "min"
    if any(w in q for w in ["correlation","relationship","effect","impact","association"]): return "correlation"
    if any(w in q for w in ["share","proportion","percentage","market share"]): return "share"
    if any(w in q for w in ["top","biggest","largest","highest","most"]): return "ranking"
    if any(w in q for w in ["season","monthly pattern","cyclical","seasonal"]): return "seasonal"
    if any(w in q for w in ["anomaly","outlier","unusual","deviation"]): return "anomaly"
    if any(w in q for w in ["ratio","dependence","share of","vs "]): return "ratio"
    if "rolling" in q or "moving average" in q: return "rolling_avg"
    if any(w in q for w in ["compare","versus","vs","difference","gap","contrast"]): return "compare"
    return "none"

def intelligent_chart_type_selection(raw_results, query: str, explicit_type: str = None):
    if explicit_type:
        return explicit_type
    if not raw_results:
        return "bar"
    q = query.lower()
    if any(w in q for w in ["trend","over time","monthly","yearly"]):
        return "line"
    if "share" in q or "composition" in q:
        return "pie"
    return "bar"

def needs_trend_clarification(query: str) -> bool:
    q = query.lower()
    if "trend" not in q:
        return False
    # if explicit years or relative ranges are included, no clarification needed
    if re.search(r'\b(19|20)\d{2}\b', q):
        return False
    if re.search(r'\b(last|past)\s+\d+\s+(year|years|month|months)\b', q):
        return False
    if ("from" in q and ("to" in q or "until" in q)) or "-" in q:
        return False
    return True

def scrub_schema_mentions(text: str) -> str:
    """Best-effort scrubber to avoid leaking table/schema terms if LLM attempts it."""
    if not text:
        return text
    for t in allowed_tables:
        text = re.sub(rf'\b{re.escape(t)}\b', "the database", text, flags=re.IGNORECASE)
    text = re.sub(r'\b(schema|table|column|sql|join)\b', 'data', text, flags=re.IGNORECASE)
    return text

# ---------- Core shaping of SQL results → chartable data ----------
def process_sql_results_for_chart(raw_results, query: str, unit: str = "Value"):
    """
    Converts a typical 2-column SQL result [(x, y), ...] into chart_data + metadata.
    If the x column looks date-like, supports optional monthly/yearly aggregation
    based on the user's wording.
    """
    chart_data = []
    metadata = {
        "title": "Energy Data Visualization",
        "xAxisTitle": "Category",
        "yAxisTitle": f"Value ({unit})",
        "datasetLabel": "Data"
    }

    df = pd.DataFrame(raw_results, columns=["date_or_cat","value"])
    df["value"] = df["value"].apply(convert_decimal_to_float).astype(float)

    q = query.lower()
    want_year = any(w in q for w in ["year", "annual", "yearly"])
    want_month = any(w in q for w in ["month", "monthly"])

    is_dt = False
    try:
        df["date_or_cat"] = pd.to_datetime(df["date_or_cat"])
        is_dt = True
    except:
        is_dt = False

    if is_dt:
        if want_year:
            df["year"] = df["date_or_cat"].dt.year
            grouped = df.groupby("year")["value"].sum().reset_index()
            chart_data = [{"date": str(row["year"]), "value": round(row["value"], 1)} for _, row in grouped.iterrows()]
            metadata["xAxisTitle"] = "Year"
        elif want_month:
            df["month"] = df["date_or_cat"].dt.to_period("M").astype(str)
            grouped = df.groupby("month")["value"].sum().reset_index()
            chart_data = [{"date": row["month"], "value": round(row["value"], 1)} for _, row in grouped.iterrows()]
            metadata["xAxisTitle"] = "Month"
        else:
            df = df.sort_values("date_or_cat")
            for _, r in df.iterrows():
                chart_data.append({"date": str(r["date_or_cat"].date()), "value": round(r["value"], 1)})
            metadata["xAxisTitle"] = "Date"
    else:
        for r in raw_results:
            chart_data.append({"sector": str(r[0]), "value": round(float(convert_decimal_to_float(r[1])), 1)})
        metadata["xAxisTitle"] = "Category"

    return chart_data, metadata

# ---------- Analyses ----------
def perform_forecast(chart_data, target="2030-12-01"):
    df = pd.DataFrame(chart_data)
    if "date" not in df or "value" not in df or df.empty:
        return None, None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if df["date"].nunique() < 2:
        return None, None
    x = (df["date"] - df["date"].min()).dt.days.values
    y = df["value"].values
    coeffs = np.polyfit(x, y, 1)
    future_x = (pd.to_datetime(target) - df["date"].min()).days
    forecast = np.polyval(coeffs, future_x)
    return round(float(forecast), 1), target

def perform_yoy(chart_data):
    df = pd.DataFrame(chart_data)
    if "date" not in df or "value" not in df or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]); df["year"] = df["date"].dt.year
    yearly = df.groupby("year")["value"].mean().sort_index()
    if len(yearly) < 2:
        return None
    last, prev = yearly.iloc[-1], yearly.iloc[-2]
    if prev == 0:
        return None
    return round((last - prev) / prev * 100, 1)

def perform_mom(chart_data):
    df = pd.DataFrame(chart_data)
    if "date" not in df or "value" not in df or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if len(df) < 2:
        return None
    prev = df.iloc[-2]["value"]
    if prev == 0:
        return None
    last = df.iloc[-1]["value"]
    return round((last - prev) / prev * 100, 1)

def perform_cagr(chart_data):
    df = pd.DataFrame(chart_data)
    if "date" not in df or "value" not in df or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date")
    if len(df) < 2:
        return None
    start, end = df.iloc[0], df.iloc[-1]
    years = (end["date"].year - start["date"].year)
    if years <= 0 or start["value"] <= 0:
        return None
    return round(((end["value"] / start["value"]) ** (1 / years) - 1) * 100, 1)

def perform_average(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    return round(df["value"].mean(), 1)

def perform_sum(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    return round(df["value"].sum(), 1)

def perform_min(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    idx = df["value"].idxmin()
    row = df.loc[idx]
    return round(float(row["value"]), 1), row.get("date") or row.get("sector")

def perform_max(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    idx = df["value"].idxmax()
    row = df.loc[idx]
    return round(float(row["value"]), 1), row.get("date") or row.get("sector")

def perform_correlation(raw_results):
    df = pd.DataFrame(raw_results)
    if df.empty:
        return None
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return None
    corr = num.corr().round(2)
    out = []
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if j <= i:
                continue
            r = corr.loc[c1, c2]
            if pd.isna(r):
                continue
            strength = "weak" if abs(r) < 0.3 else ("moderate" if abs(r) < 0.6 else "strong")
            direction = "positive" if r > 0 else "negative"
            out.append(f"{c1} vs {c2}: {r} ({strength}, {direction})")
    return "\n".join(out) if out else None

def perform_share(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    total = df["value"].sum()
    if total == 0:
        return None
    label_col = "sector" if "sector" in df.columns else "date"
    return {str(row[label_col]): round(row["value"] / total * 100, 1) for _, row in df.iterrows()}

def perform_ranking(chart_data, n=3):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    label_col = "sector" if "sector" in df.columns else "date"
    df = df.sort_values("value", ascending=False)
    top = df[[label_col, "value"]].head(n)
    return [{label_col: str(r[label_col]), "value": round(float(r["value"]), 1)} for _, r in top.iterrows()]

def perform_seasonal(chart_data):
    df = pd.DataFrame(chart_data)
    if "date" not in df or "value" not in df or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    avg = df.groupby("month")["value"].mean().round(1)
    return {int(k): float(v) for k, v in avg.to_dict().items()}

def perform_anomaly(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    m, sd = df["value"].mean(), df["value"].std()
    if sd == 0 or np.isnan(sd):
        return []
    anomalies = df[(df["value"] > m + 2 * sd) | (df["value"] < m - 2 * sd)]
    return anomalies.to_dict(orient="records")

def perform_ratio(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty or len(df) < 2:
        return None
    a, b = df.iloc[-1]["value"], df.iloc[-2]["value"]
    if b == 0:
        return None
    return round(a / b, 2)

def perform_rolling_avg(chart_data, window=3):
    df = pd.DataFrame(chart_data)
    if "date" not in df or "value" not in df or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["rolling"] = df["value"].rolling(window).mean()
    rolled = df.dropna(subset=["rolling"])[["date", "rolling"]]
    return [{"date": str(r["date"].date()), "value": round(float(r["rolling"]), 1)} for _, r in rolled.iterrows()]

def perform_comparison(chart_data):
    df = pd.DataFrame(chart_data)
    if df.empty or "value" not in df:
        return None
    label_col = "sector" if "sector" in df.columns else ("date" if "date" in df.columns else None)
    if label_col is None:
        return None
    if label_col == "date":
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        if len(df) < 2:
            return None
        a, b = df.iloc[-1], df.iloc[-2]
    else:
        df = df.sort_values("value", ascending=False)
        if len(df) < 2:
            return None
        a, b = df.iloc[0], df.iloc[1]
    val_a, val_b = float(a["value"]), float(b["value"])
    abs_diff = round(val_a - val_b, 1)
    pct_diff = round((val_a - val_b) / val_b * 100, 1) if val_b != 0 else None
    ratio = round(val_a / val_b, 2) if val_b != 0 else None
    return {
        "group_a": str(a[label_col]),
        "group_b": str(b[label_col]),
        "val_a": round(val_a, 1),
        "val_b": round(val_b, 1),
        "abs_diff": abs_diff,
        "pct_diff": pct_diff,
        "ratio": ratio
    }

# ---------- Endpoints ----------
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
        is_chart, chart_type = is_chart_request(q.query)
        analysis_type = detect_analysis_type(q.query)
        unit = detect_unit(q.query)

        # If a trend is requested without any period hint, ask for clarification first.
        if analysis_type == "trend" and needs_trend_clarification(q.query):
            msg = "Do you want the full 2015–2025 trend, or a specific period (e.g., 2020–2024 or last 3 years)?"
            return {"answer": msg, "chartType": None, "data": None, "chartMetadata": None}

        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="openai-tools",
            system_message=SYSTEM_PROMPT
        )

        # Run agent and request intermediate steps
        result = agent.invoke({"input": q.query}, return_intermediate_steps=True)
        response_text = result.get("output", "")
        steps = result.get("intermediate_steps", [])

        if steps:
            try:
                last_step = steps[-1]
                sql_cmd = None
                if isinstance(last_step, tuple) and "sql_cmd" in last_step[0]:
                    sql_cmd = last_step[0]["sql_cmd"]
                elif isinstance(last_step, dict) and "sql_cmd" in last_step:
                    sql_cmd = last_step["sql_cmd"]

                if sql_cmd:
                    with engine.connect() as conn:
                        raw = conn.execute(text(clean_sql(sql_cmd))).fetchall()
                        if raw:
                            chart_data, meta = process_sql_results_for_chart(raw, q.query, unit)
                            chart_type_opt = intelligent_chart_type_selection(raw, q.query, chart_type)

                            # ----- Apply requested analysis -----
                            note = ""
                            if analysis_type == "forecast":
                                f, t = perform_forecast(chart_data)
                                if f is not None:
                                    note = f"\n📈 Forecast for {t}: {format_number(f, unit)} (approximate)"
                            elif analysis_type == "yoy":
                                c = perform_yoy(chart_data)
                                if c is not None:
                                    note = f"\n📊 YoY change: {c}%"
                            elif analysis_type == "mom":
                                c = perform_mom(chart_data)
                                if c is not None:
                                    note = f"\n📊 MoM change: {c}%"
                            elif analysis_type == "cagr":
                                c = perform_cagr(chart_data)
                                if c is not None:
                                    note = f"\n📈 CAGR: {c}%"
                            elif analysis_type == "average":
                                avg = perform_average(chart_data)
                                if avg is not None:
                                    note = f"\n🔢 Average: {format_number(avg, unit)}"
                            elif analysis_type == "sum":
                                total = perform_sum(chart_data)
                                if total is not None:
                                    note = f"\n🔢 Total: {format_number(total, unit)}"
                            elif analysis_type == "max":
                                res = perform_max(chart_data)
                                if res:
                                    val, when = res
                                    note = f"\n📈 Max: {format_number(val, unit)} on {when}"
                            elif analysis_type == "min":
                                res = perform_min(chart_data)
                                if res:
                                    val, when = res
                                    note = f"\n📉 Min: {format_number(val, unit)} on {when}"
                            elif analysis_type == "correlation":
                                cor = perform_correlation(raw)
                                if cor:
                                    note = f"\n🔗 Correlation analysis:\n{cor}"
                            elif analysis_type == "share":
                                s = perform_share(chart_data)
                                if s:
                                    note = f"\n📊 Shares (% of total): {s}"
                            elif analysis_type == "ranking":
                                r = perform_ranking(chart_data)
                                if r:
                                    note = f"\n🏆 Top values: {r}"
                            elif analysis_type == "seasonal":
                                s = perform_seasonal(chart_data)
                                if s:
                                    note = f"\n📅 Seasonal pattern (avg by month): {s}"
                            elif analysis_type == "anomaly":
                                a = perform_anomaly(chart_data)
                                if a:
                                    note = f"\n⚠️ Anomalies (±2σ): {a}"
                            elif analysis_type == "ratio":
                                r = perform_ratio(chart_data)
                                if r:
                                    note = f"\n➗ Simple ratio (last/prev): {r}"
                            elif analysis_type == "rolling_avg":
                                r = perform_rolling_avg(chart_data)
                                if r:
                                    note = f"\n📉 Rolling average (3): {r}"
                            elif analysis_type == "compare":
                                comp = perform_comparison(chart_data)
                                if comp:
                                    note = (
                                        f"\n🔍 Comparison:\n"
                                        f"{comp['group_a']} = {format_number(comp['val_a'], unit)}, "
                                        f"{comp['group_b']} = {format_number(comp['val_b'], unit)} → "
                                        f"Δ: {format_number(comp['abs_diff'], unit)}"
                                    )
                                    if comp['pct_diff'] is not None:
                                        note += f" ({comp['pct_diff']}%)"
                                    if comp['ratio'] is not None:
                                        note += f", ratio: {comp['ratio']}"

                            # Build a clean text list of values
                            lines = [f"- {r[0]}: {format_number(convert_decimal_to_float(r[1]), unit)}" for r in raw]
                            ans = "Here’s the requested data:\n\n" + "\n".join(lines) + note

                            # Final scrub against schema leakage (belt & suspenders)
                            ans = scrub_schema_mentions(ans)

                            return {
                                "answer": ans,
                                "chartType": chart_type_opt if is_chart else None,
                                "data": chart_data if is_chart else None,
                                "chartMetadata": meta if is_chart else None
                            }
            except Exception as e:
                print("❌ SQL/analysis error:", e)
                safe_text = scrub_schema_mentions(response_text)
                return {"answer": safe_text, "chartType": None, "data": None, "chartMetadata": None}

        # Fallback: text only
        response_text = scrub_schema_mentions(response_text)
        return {"answer": response_text, "chartType": None, "data": None, "chartMetadata": None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
