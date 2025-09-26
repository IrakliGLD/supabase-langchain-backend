# === context.py v1.4 ===
# Unified schema doc + joins + human-friendly labels + scrubber

import re

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
    return text.replace("```", "").strip()


# === DB_SCHEMA_DOC v1.4 ===
DB_SCHEMA_DOC = """
### Global Rules & Conversions ###
- **General Rule:** Provide summaries and insights only. Do NOT return raw data, full tables, or row-level dumps. If asked for a dump, refuse and suggest an aggregated view instead.
- **Unit Conversion:** To compare data between tables, use:
  - 1 TJ = 277.778 MWh
  - The `tech_quantity` and `trade` tables store quantities in **thousand MWh**; multiply by 1000 for MWh.
- **Data Granularity:** All tables with a `date` column contain **monthly** data (first day of month).
- **Timeframe:** Data generally spans from 2015 to present.

---
### Table: public.dates ###
Central reference for all monthly records.
**Columns:**
- `date`: Month (YYYY-MM-DD, first day).

---
### Table: public.energy_balance_long ###
Energy consumption by sector and source (GEOSTAT).
**Columns:**
- `year`: Calendar year.
- `sector`: Consuming sector (Industry, Transport, Other use).
- `energy_source`: Energy type (Coal, Oil products, Natural Gas, Hydro, Wind, Solar, Biofuel & Waste, Electricity, Heat, Total).
- `volume_tj`: Consumption in **TJ**.

---
### Table: public.entities ###
Power sector entities.
**Columns:**
- `entity`: Raw name.
- `entity_normalized`: Standardized identifier **(use for joins/analysis)**.
- `type`: Entity type (HPP, TPP, Solar, Wind).
- `ownership`: Owner.
- `source`: Local vs import-dependent.

---
### Table: public.monthly_cpi ###
CPI (2015=100).
**Columns:**
- `date`: Month.
- `cpi_type`: CPI category (e.g., overall CPI, electricity_gas_and_other_fuels).
- `cpi`: Value.
**Note:** Large drop in electricity_gas_and_other_fuels CPI in Dec 2020–Feb 2021 due to subsidies.

---
### Table: public.price ###
Monthly electricity prices.
**Columns:**
- `date`: Month.
- `p_dereg_gel`: Deregulated price (GEL/MWh).
- `p_bal_gel`: **Balancing price** (GEL/MWh).
- `p_gcap_gel`: Guaranteed capacity fee (GEL/MWh).
- `xrate`: GEL/USD.
- `p_dereg_usd`, `p_bal_usd`, `p_gcap_usd`: USD equivalents.

---
### Table: public.tariff_gen ###
GNERC-regulated tariffs.
**Columns:**
- `date`: Month.
- `entity`: Plant (join with entities.entity).
- `tariff_gel`: Tariff (GEL/MWh).
- `tariff_usd`: Tariff (USD/MWh).
**Rules:**
- Missing **thermal** tariff ⇒ no generation that month.
- Missing **hydro** tariff starting a month ⇒ deregulated onward.

---
### Table: public.tech_quantity ###
Generation, demand, trade/transit aggregates.
**Columns:**
- `date`: Month.
- `type_tech`: Category (`hydro`, `thermal`, `wind`, `import`, `export`, `losses`, `abkhazeti`, `transit`).
- `quantity_tech`: **Thousand MWh**.
**Key Synonyms for `type_tech`:**
- "hydro generation", "HPP" → `hydro`
- "thermal generation", "TPP" → `thermal`
- "wind generation", "wind power plant", "wind turbine" → `wind`
- "imports" → `import`
- "exports" → `export`
- "losses" → `losses`
- "Abkhazeti consumption" → `abkhazeti`
- "transit" → `transit` (spiked in 2022–2023 alongside Turkey gas prices)

---
### Table: public.trade ###
Electricity trade outcomes by **entity** and **segment**.
**Columns:**
- `date`: Month.
- `entity`: Plant (join with entities.entity).
- `segment`: Market segment (`balancing`, `bilateral`, `exchange`, `thermal_ppa`, `renewable_ppa`).
- `quantity`: **Thousand MWh**.
**Key Synonyms for `segment`:**
- "balancing electricity", "balancing market" → `balancing`
- "bilateral contracts", "direct contracts" → `bilateral`
- "thermal PPA", "conventional PPA" → `thermal_ppa`
- "renewable PPA", "RES PPA", "green PPA" → `renewable_ppa`
**Analytical Rules:**
- **Renewable PPA share in balancing** (monthly):
  - numerator: sum `quantity` where `segment = 'renewable_ppa'`
  - denominator: sum `quantity` where `segment = 'balancing'`
  - share = numerator / denominator
- **PPA participation**: (`thermal_ppa + renewable_ppa`) vs total balancing volume.
**Notes:**
- Bilateral and exchange can be summed in some reports.
- PPAs must sell to ESCO during mandatory periods.
- We do not have separation of trade on exchange and with bilateral contract. THere are two segments: balacing electricity (so called balancing makret) and Bilateral & Exchange in total.
- The exchange was launched in july 2024.

---
### Cross-Table Analytical Logic ###
- **Price ↔ Trade**: Join on `date`. Compare `p_bal_gel` vs PPA shares in balancing.
- **Price ↔ Tech Quantity**: Join on `date`. Supply/demand vs price dynamics.
- **Price ↔ Monthly CPI**: Join on `date`. Price vs inflation categories.
- **Price ↔ Energy Balance**: Aggregate price monthly to annual, compare to use.
- **Trade ↔ Tech Quantity**: Join on `date`. Traded vs generated/exported.
- **Trade ↔ Tariff Gen**: Join on `date` & `entity`. Tariff signals vs participation.
- **Trade ↔ Entities**: Join on `entity`. Enrich trade with entity metadata.
- **Trade ↔ Energy Balance**: Aggregate trade (annual) vs sector demand.
- **Tech Quantity ↔ Energy Balance**: Annual totals vs sectoral consumption.
- **Tech Quantity ↔ Monthly CPI**: Join on `date`. Demand/supply vs CPI.
- **Tariff Gen ↔ Entities**: Join on `entity` for metadata.
- **Tariff Gen ↔ Price**: Join on `date`. Regulated vs market indicators.
- **Tariff Gen ↔ Tech Quantity**: Join on `date` & `entity`. Tariffs vs output.

"""

# === DB_JOINS v1.4 ===
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
