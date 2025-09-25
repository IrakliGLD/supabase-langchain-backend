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
The central reference table for all monthly records.

**Columns:**
- `date`: The month of the record (YYYY-MM-DD, typically first day of the month).

---
### Table: public.energy_balance_long ###
Shows the energy consumption of different sectors, based on GEOSTAT data.

**Columns:**
- `year`: The calendar year of the record.
- `sector`: The consuming sector.
- `energy_source`: The type of energy consumed (Coal, Oil, Gas, Hydro, Wind, Solar, Biofuel & Waste, Electricity, Heat, Total).
- `volume_tj`: Energy consumption in **TJ**.

---
### Table: public.entities ###
Defines entities in the electricity sector.

**Columns:**
- `entity`: Raw entity name.
- `entity_normalized`: The unique, standardized identifier (use this for joins/analysis).
- `type`: Entity type (HPP, TPP, Solar, Wind).
- `ownership`: Owner of the entity.
- `source`: Local or import-dependent.

---
### Table: public.monthly_cpi ###
Monthly Consumer Price Index (CPI), indexed to 2015=100.

**Columns:**
- `date`: Month of record.
- `cpi_type`: CPI category (e.g., overall CPI, electricity_gas_and_other_fuels).
- `cpi`: Numeric CPI value.

---
### Table: public.price ###
Contains monthly electricity market prices.

**Columns:**
- `date`: Month of record.
- `p_dereg_gel`: Price for deregulated plants (GEL/MWh).
- `p_bal_gel`: Balancing price (GEL/MWh).
- `p_gcap_gel`: Guaranteed capacity fee (GEL/MWh).
- `xrate`: GEL/USD exchange rate.
- `p_dereg_usd`, `p_bal_usd`, `p_gcap_usd`: USD equivalents.

---
### Table: public.tariff_gen ###
Regulated tariffs approved by GNERC.

**Columns:**
- `date`: Month of record.
- `entity`: Name of the generating unit (join to entities.entity).
- `tariff_gel`: Regulated tariff in GEL/MWh.
- `tariff_usd`: Regulated tariff in USD/MWh.

**Rules:**
- Thermal plant: missing tariff → did not generate that month.
- Hydro plant: missing tariff → deregulated from that month onward.

---
### Table: public.tech_quantity ###
Total electricity demand, supply, imports/exports.

**Columns:**
- `date`: Month of record.
- `type_tech`: Category (hydro, thermal, wind, import, export, losses, abkhazeti, transit).
- `quantity_tech`: Amount in **thousand MWh**.

---
### Table: public.trade ###
Market outcomes by entity and segment.

**Columns:**
- `date`: Month of record.
- `entity`: Raw entity name (join to entities.entity).
- `segment`: Market segment.
- `quantity`: Traded energy in **thousand MWh**.

**Notes:**
- Bilateral and exchange summed together.
- PPA entities (thermal_ppa, renewable_ppa) must sell all output to ESCO during mandatory periods.

---
### Cross-Table Analytical Logic ###

- **Price ↔ Trade**
  - Join on `date`.
  - Study how prices correlate with traded volumes in specific segments.
  - Example: Compare `p_bal_gel` with share of PPAs (`thermal_ppa + renewable_ppa`) in balancing trade.

- **Price ↔ Tech Quantity**
  - Join on `date`.
  - Analyze how generation categories (hydro, thermal, wind, imports) influence prices.
  - Example: Hydro generation vs. balancing price.

- **Price ↔ Monthly CPI**
  - Join on `date`.
  - Correlate electricity prices with CPI categories.
  - Example: `p_bal_gel` vs. CPI for electricity/gas/fuels.

- **Price ↔ Energy Balance**
  - Aggregate `energy_balance_long` by year and compare to yearly average prices.

- **Trade ↔ Tech Quantity**
  - Join on `date`.
  - Compare traded volumes with total generation/consumption.
  - Example: exports in `tech_quantity` vs. exports in `trade`.

- **Trade ↔ Tariff Gen**
  - Join on `date` and `entity`.
  - See how regulated tariffs affect market participation.

- **Trade ↔ Entities**
  - Join on `entity`.
  - Enrich trade data with metadata (ownership, type, local/import dependent).

- **Trade ↔ Energy Balance**
  - Aggregate trade yearly (date→year) and join with `energy_balance_long.year`.

- **Tech Quantity ↔ Energy Balance**
  - Aggregate tech quantities to yearly totals and compare with `energy_balance_long`.

- **Tech Quantity ↔ Monthly CPI**
  - Join on `date`.
  - Correlate demand/supply dynamics with inflation.

- **Tariff Gen ↔ Entities**
  - Join on `entity`.
  - Link tariffed plants with type/ownership metadata.

- **Tariff Gen ↔ Price**
  - Join on `date`.
  - Compare regulated tariffs with market price levels.

- **Tariff Gen ↔ Tech Quantity**
  - Join on `date` and `entity` (if applicable).
  - Compare regulated capacity with generation output.

- **Energy Balance ↔ Monthly CPI**
  - Aggregate energy balance yearly and compare with yearly CPI averages.

- **Entities ↔ Price**
  - No direct join; use trade as intermediary:
    - `entities → trade → price`.
"""
