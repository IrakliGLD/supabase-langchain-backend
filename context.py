# === context.py v1.3 ===
# Holds schema documentation and join rules

DB_SCHEMA_DOC = """
### Global Rules & Conversions ###
- **General Rule:** You must only provide summaries and insights. You are NOT allowed to return raw data, full tables, or detailed row-level datasets. If a user asks for a data dump, you must refuse and suggest an aggregated view instead.
- **Unit Conversion:** To compare data between tables, use these exact factors:
  - 1 TJ = 277.778 MWh
  - The `tech_quantity` and `trade` tables are in "thousand MWh". To compare with a value in MWh, multiply their `quantity` values by 1000.
- **Data Granularity:** All tables with a `date` column contain **monthly** data. The date typically represents the first day of the month.
- **Timeframe:** Data generally spans from 2015 to the present.

---
### Table: public.dates ###
Central reference for all monthly records.

**Columns:**
- `date`: Month of the record (YYYY-MM-DD, typically first day of the month).

---
### Table: public.energy_balance_long ###
Energy consumption of different sectors (GEOSTAT data).

**Columns:**
- `year`: Calendar year.
- `sector`: Consuming sector (Industry, Transport, Other use).
- `energy_source`: Energy type (Coal, Oil, Gas, Hydro, Wind, Solar, Biofuel & Waste, Electricity, Heat, Total).
- `volume_tj`: Consumption in TJ.

---
### Table: public.entities ###
Power sector entities.

**Columns:**
- `entity`: Raw name.
- `entity_normalized`: Standardized identifier (use for joins/analysis).
- `type`: Entity type (HPP, TPP, Solar, Wind).
- `ownership`: Owner.
- `source`: Local or import-dependent.

---
### Table: public.monthly_cpi ###
Consumer Price Index, 2015=100.

**Columns:**
- `date`: Month.
- `cpi_type`: CPI category (overall, electricity_gas_and_other_fuels, etc.).
- `cpi`: Value.

---
### Table: public.price ###
Monthly electricity prices.

**Columns:**
- `date`: Month.
- `p_dereg_gel`: Deregulated price (GEL/MWh).
- `p_bal_gel`: Balancing price (GEL/MWh).
- `p_gcap_gel`: Guaranteed capacity fee (GEL/MWh).
- `xrate`: GEL/USD.
- `p_dereg_usd`, `p_bal_usd`, `p_gcap_usd`: USD equivalents.

---
### Table: public.tariff_gen ###
GNERC-regulated tariffs.

**Columns:**
- `date`: Month.
- `entity`: Plant name (join with entities.entity).
- `tariff_gel`: Tariff in GEL/MWh.
- `tariff_usd`: Tariff in USD/MWh.

**Rules:**
- Missing thermal tariff = no generation.
- Missing hydro tariff = deregulated.

---
### Table: public.tech_quantity ###
Electricity demand, supply, imports/exports.

**Columns:**
- `date`: Month.
- `type_tech`: Category (hydro, thermal, wind, import, export, losses, abkhazeti, transit).
- `quantity_tech`: Thousand MWh.

---
### Table: public.trade ###
Market outcomes.

**Columns:**
- `date`: Month.
- `entity`: Plant name (join with entities.entity).
- `segment`: Market segment (balancing, bilateral, exchange, thermal_ppa, renewable_ppa).
- `quantity`: Thousand MWh.

**Key Synonyms for `segment`:**
- "balancing electricity", "balancing market" → `segment = 'balancing'`
- "bilateral contracts", "direct contracts" → `segment = 'bilateral'`
- "exchange market", "market platform" → `segment = 'exchange'`
- "thermal PPA", "conventional PPA" → `segment = 'thermal_ppa'`
- "renewable PPA", "RES PPA", "green PPA" → `segment = 'renewable_ppa'`

**Important Analytical Logic:**
- To compute the **share of renewable PPA in balancing electricity**:
  1. Aggregate `quantity` where `segment = 'renewable_ppa'`.
  2. Divide by total `quantity` where `segment = 'balancing'` for the same period.
- To study **PPA participation**: compare `thermal_ppa + renewable_ppa` against total balancing volumes.
- To study **market split**: compare shares of `bilateral`, `exchange`, `balancing`, and `PPA`.

**Notes:**
- Bilateral and exchange are currently summed together in some official stats.
- PPAs (thermal_ppa, renewable_ppa) must sell to ESCO during mandatory periods.

---
### Cross-Table Analytical Logic ###

- **Price ↔ Trade**: Join on `date`. Compare `p_bal_gel` vs. PPA share in balancing.
- **Price ↔ Tech Quantity**: Join on `date`. Analyze generation impact on price.
- **Price ↔ Monthly CPI**: Join on `date`. Correlate price vs. inflation.
- **Price ↔ Energy Balance**: Yearly aggregation.
- **Trade ↔ Tech Quantity**: Join on `date`. Compare traded vs. generated/exported.
- **Trade ↔ Tariff Gen**: Join on `date` & `entity`. Tariffs vs. participation.
- **Trade ↔ Entities**: Join on `entity`. Enrich trade with metadata.
- **Trade ↔ Energy Balance**: Aggregate trade yearly; compare with sector demand.
- **Tech Quantity ↔ Energy Balance**: Yearly totals vs. sectoral consumption.
- **Tech Quantity ↔ Monthly CPI**: Join on `date`. Demand/supply vs. inflation.
- **Tariff Gen ↔ Entities**: Join on `entity`. Add plant metadata.
- **Tariff Gen ↔ Price**: Join on `date`. Compare regulated vs. market.
- **Tariff Gen ↔ Tech Quantity**: Join on `date` and `entity`. Tariff vs. output.
- **Energy Balance ↔ Monthly CPI**: Yearly comparison demand vs. inflation.
- **Entities ↔ Price**: Indirect via trade: entities → trade → price.
"""

# --- DB_JOINS v1.3 ---
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
