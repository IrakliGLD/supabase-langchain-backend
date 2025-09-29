# === context.py v1.7 ===
# Changes from v1.6: Added `import os` to fix NameError for os.getenv. No other changes—kept hybrid scrub, schema dict, prose doc, labels, etc. Realistic note: Fix resolves deployment error; no impact on latency/cost since os is standard library.


import os  # Added to fix NameError
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Column label mapping ---
COLUMN_LABELS = {
    # shared
    "date": "Period (Year-Month)",

    # energy_balance_long
    "year": "Year",
    "sector": "Sector",
    "energy_source": "Energy Source",
    "volume_tj": "Energy Consumption (TJ)",

    # entities
    "entity": "Entity Name",
    "entity_normalized": "Standardized Entity ID",
    "type": "Entity Type",
    "ownership": "Ownership",
    "source": "Source (Local vs Import-Dependent)",

    # monthly_cpi
    "cpi_type": "CPI Category",
    "cpi": "CPI Value (2015=100)",

    # price
    "p_dereg_gel": "Deregulated Price (GEL/MWh)",
    "p_bal_gel": "Balancing Price (GEL/MWh)",
    "p_gcap_gel": "Guaranteed Capacity Fee (GEL/MWh)",
    "xrate": "Exchange Rate (GEL/USD)",
    "p_dereg_usd": "Deregulated Price (USD/MWh)",
    "p_bal_usd": "Balancing Price (USD/MWh)",
    "p_gcap_usd": "Guaranteed Capacity Fee (USD/MWh)",

    # tariff_gen
    "tariff_gel": "Regulated Tariff (GEL/MWh)",
    "tariff_usd": "Regulated Tariff (USD/MWh)",

    # tech_quantity
    "type_tech": "Technology Type",
    "quantity_tech": "Quantity (thousand MWh)",

    # trade
    "segment": "Market Segment",
    "quantity": "Trade Volume (thousand MWh)",
}

# --- Table label mapping ---
TABLE_LABELS = {
    "dates": "Calendar (Months)",
    "energy_balance_long": "Energy Balance (by Sector)",
    "entities": "Power Sector Entities",
    "monthly_cpi": "Consumer Price Index (CPI)",
    "price": "Electricity Market Prices",
    "tariff_gen": "Regulated Tariffs",
    "tech_quantity": "Generation & Demand Quantities",
    "trade": "Electricity Trade",
}

# --- Value label mapping ---
VALUE_LABELS = {
    # tech_quantity.type_tech
    "hydro": "Hydro Generation",
    "thermal": "Thermal Generation",
    "wind": "Wind Generation",
    "solar": "Solar Generation",
    "import": "Imports",
    "export": "Exports",
    "losses": "Grid Losses",
    "abkhazeti": "Abkhazia Consumption",
    "transit": "Transit Flows",

    # entities.type
    "HPP": "Hydropower Plant",
    "TPP": "Thermal Power Plant",
    "Solar": "Solar Plant",
    "Wind": "Wind Plant",
    "Import": "Import",

    # CPI categories
    "overall CPI": "Overall Consumer Price Index",
    "electricity_gas_and_other_fuels": "Electricity, Gas & Other Fuels CPI",

    # energy sources in energy_balance_long
    "Coal": "Coal",
    "Oil products": "Oil Products",
    "Natural Gas": "Natural Gas",
    "Hydro": "Hydropower",
    "Wind": "Wind Power",
    "Solar": "Solar Power",
    "Biofuel & Waste": "Biofuel & Waste",
    "Electricity": "Electricity",
    "Heat": "Heat",
    "Total": "Total Energy Use",

    # trade.segment values
    "balancing_electricity": "Balancing Electricity",
    "bilateral_exchange": "Bilateral Contracts & Exchange",
    "renewable_ppa": "Renewable PPA",
    "thermal_ppa": "Thermal PPA",
}

# --- Structured Schema Dict (for validation) ---
DB_SCHEMA_DICT = {
    "tables": {
        "dates": {"columns": ["date"], "desc": "Calendar (Months)"},
        "energy_balance_long": {"columns": ["year", "sector", "energy_source", "volume_tj"], "desc": "Energy Balance (by Sector)"},
        "entities": {"columns": ["entity", "entity_normalized", "type", "ownership", "source"], "desc": "Power Sector Entities"},
        "monthly_cpi": {"columns": ["date", "cpi_type", "cpi"], "desc": "Consumer Price Index (CPI)"},
        "price": {"columns": ["date", "p_dereg_gel", "p_bal_gel", "p_gcap_gel", "xrate", "p_dereg_usd", "p_bal_usd", "p_gcap_usd"], "desc": "Electricity Market Prices"},
        "tariff_gen": {"columns": ["date", "entity", "tariff_gel", "tariff_usd"], "desc": "Regulated Tariffs"},
        "tech_quantity": {"columns": ["date", "type_tech", "quantity_tech"], "desc": "Generation & Demand Quantities"},
        "trade": {"columns": ["date", "entity", "segment", "quantity"], "desc": "Electricity Trade"},
    },
    "rules": {
        "unit_conversion": "1 TJ = 277.778 MWh; tech_quantity/trade in thousand MWh (multiply by 1000 for MWh).",
        "granularity": "Monthly data (first day of month).",
        "timeframe": "2015 to present.",
        "forecast_restriction": "Only for prices, CPI, demand; no generation/imports/exports."
    }
}

# --- Prose Schema Doc (for LLM context) ---
DB_SCHEMA_DOC = """
### Global Rules & Conversions ###
- **General Rule:** Provide summaries and insights only. Do NOT return raw data, full tables, or row-level dumps. If asked for a dump, refuse and suggest an aggregated view instead.
- **Unit Conversion:** To compare data between tables, use:
  - 1 TJ = 277.778 MWh
  - The `tech_quantity` and `trade` tables store quantities in **thousand MWh**; multiply by 1000 for MWh.
- **Data Granularity:** All tables with a `date` column contain **monthly** data (first day of month).
- **Timeframe:** Data generally spans from 2015 to present.
- **Forecasting Restriction:** Forecasts can be made for prices, CPI, and demand. 
  For generation (hydro, thermal, wind, solar) and imports/exports: 
  only historical trends can be shown. Future projections depend on new capacity/projects not available in this data.

---
### Table: public.dates ###
- Columns: date
- Description: Calendar data with monthly timestamps.

### Table: public.energy_balance_long ###
- Columns: year, sector, energy_source, volume_tj
- Description: Annual energy consumption by sector and source, in terajoules (TJ).

### Table: public.entities ###
- Columns: entity, entity_normalized, type, ownership, source
- Description: Details on power sector entities (e.g., power plants, importers).

### Table: public.monthly_cpi ###
- Columns: date, cpi_type, cpi
- Description: Monthly Consumer Price Index (CPI) data, base year 2015=100.

### Table: public.price ###
- Columns: date, p_dereg_gel, p_bal_gel, p_gcap_gel, xrate, p_dereg_usd, p_bal_usd, p_gcap_usd
- Description: Monthly electricity market prices in GEL and USD per MWh.

### Table: public.tariff_gen ###
- Columns: date, entity, tariff_gel, tariff_usd
- Description: Monthly regulated tariffs for entities, in GEL and USD per MWh.

### Table: public.tech_quantity ###
- Columns: date, type_tech, quantity_tech
- Description: Monthly generation and demand quantities by technology type, in thousand MWh.

### Table: public.trade ###
- Columns: date, entity, segment, quantity
- Description: Monthly electricity trade volumes by market segment, in thousand MWh.
"""

# --- DB_JOINS v1.4 ---
DB_JOINS = {
    "dates": {"join_on": "date", "related_to": ["price", "monthly_cpi", "tech_quantity", "trade", "tariff_gen"]},
    "energy_balance_long": {"join_on": "year", "related_to": ["tech_quantity", "price", "monthly_cpi", "trade"]},
    "price": {"join_on": "date", "related_to": ["trade", "tech_quantity", "monthly_cpi", "energy_balance_long"]},
    "trade": {"join_on": ["date", "entity"], "related_to": ["price", "entities", "tariff_gen", "tech_quantity", "energy_balance_long"]},
    "tech_quantity": {"join_on": "date", "related_to": ["price", "energy_balance_long", "trade", "monthly_cpi"]},
    "tariff_gen": {"join_on": ["date", "entity"], "related_to": ["entities", "trade", "price", "tech_quantity"]},
    "entities": {"join_on": "entity", "related_to": ["tariff_gen", "trade"]},
    "monthly_cpi": {"join_on": "date", "related_to": ["price", "tech_quantity", "energy_balance_long"]},
}

# --- Human-friendly scrubber (labels-aware) ---
def scrub_schema_mentions(text: str) -> str:
    """
    Cleans final model output so that:
    - Raw SQL/schema terms are humanized.
    - Column names -> user-friendly labels.
    - Table names -> user-friendly labels.
    - Encoded categorical values -> natural labels.
    """
    if not text:
        return text

    # 1) Columns -> labels
    for col, label in COLUMN_LABELS.items():
        text = re.sub(rf"\b{re.escape(col)}\b", label, text, flags=re.IGNORECASE)

    # 2) Tables -> labels
    for tbl, label in TABLE_LABELS.items():
        text = re.sub(rf"\b{re.escape(tbl)}\b", label, text, flags=re.IGNORECASE)

    # 3) Encoded values -> natural labels
    for val, label in VALUE_LABELS.items():
        text = re.sub(rf"\b{re.escape(val)}\b", label, text, flags=re.IGNORECASE)

    # 4) Hide schema/SQL jargon
    schema_terms = [
        "schema", "table", "column", "sql", "join",
        "primary key", "foreign key", "view", "constraint"
    ]
    for term in schema_terms:
        text = re.sub(rf"\b{re.escape(term)}\b", "data", text, flags=re.IGNORECASE)

    # 5) Strip markdown fences
    text = text.replace("```", "").strip()

    # 6) LLM fallback if terms still present
    if any(re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE) for term in schema_terms):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", "Remove any technical jargon like schema, table, sql, join from text. Keep meaning intact."),
            ("human", text),
        ])
        chain = fallback_prompt | llm | StrOutputParser()
        text = chain.invoke({})

    return text
