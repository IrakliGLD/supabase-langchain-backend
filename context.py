DB_SCHEMA_DOC = """
### Global Rules & Conversions ###
- **General Rule:** You must only provide summaries and insights. You are NOT allowed to return raw data, full tables, or detailed row-level datasets. If a user asks for a data dump, you must refuse and suggest an aggregated view instead.
- **Unit Conversion:** To compare data between tables, use these exact factors:
  - 1 TJ = 277.778 MWh
  - The `tech_quantity` and `trade` tables are in "thousand MWh". To compare with a value in MWh, multiply their `quantity` values by 1000.

---
### Table: public.energy_balance_long ###
Shows the energy consumption of different sectors, based on GEOSTAT data.

**Columns (Explicit Mapping):**
- `year`: The calendar year of the record.
- `sector`: The energy-consuming sector.
  - `Industry`: Includes Iron and steel, Chemical, Non-metallic minerals, Mining, Food/beverages, Construction.
  - `Transport`: Includes Road, Rail, Pipeline transport.
  - `Other use`: Includes Commercial/public services, Residential.
- `energy_source`: The type of energy consumed (e.g., Coal, Oil products, Natural Gas, Hydro, Wind, Solar, Biofuel & Waste, Electricity, Heat). "Total" is the sum of all sources.
- `volume_tj`: Energy consumption measured in **TJ**.

---
### Table: public.entities ###
Defines the entities in the electricity sector.

**Columns (Explicit Mapping):**
- `entity`: The common name of the entity.
- `entity_normalized`: The unique, standardized identifier for an entity. **Always use this column for analysis and joins**, as `entity` names can vary.
- `type`: The classification of the entity.
- `ownership`: The owner of the entity.
- `source`: Indicates if the entity is local (`local`) or dependent on imports (`import_dependent`).

**Key Synonyms for `type` column:**
- User says "HPP", "hydro plant" -> Query for `type = 'HPP'`
- User says "TPP", "thermal plant" -> Query for `type = 'TPP'`
- User says "Solar plant" -> Query for `type = 'Solar'`
- User says "Wind plant" -> Query for `type = 'Wind'`

---
### Table: public.monthly_cpi ###
Shows the monthly Consumer Price Index (CPI), indexed to 2015=100.

**Columns (Explicit Mapping):**
- `date`: The month of the record. if you need to aggregate by month or year, use the date as it also contains information about the month and year.
- `cpi_type`: The category of the index (e.g., `overall CPI`, `electricity_gas_and_other_fuels`).
- `cpi`: The numeric CPI value.
- **Note:** A large drop in `electricity_gas_and_other_fuels` CPI occurred between Dec 2020 and Feb 2021 due to government subsidies.

---
### Table: public.price ###
Contains monthly electricity market prices.

**Columns (Explicit Mapping):**
- `date`: The month of the record. if you need to aggregate by month or year, use the date as it also contains information about the month and year.
- `p_dereg_gel`: Price (GEL/MWH) for deregulated power plants selling as balancing electricity.
- `p_bal_gel`: **Balancing electricity price** (GEL/MWH). This is the weighted average price of all balancing sales and a key market indicator.
- `p_gcap_gel`: Guaranteed capacity fee (GEL/MWH) paid by consumers to cover fixed costs of essential thermal power plants.
- `xrate`: The GEL to USD exchange rate.
- `p_dereg_usd`, `p_bal_usd`, `p_gcap_usd`: The USD equivalents of the GEL prices.

---
### Table: public.tariff_gen ###
Shows the regulated tariffs for power plants, approved by GNERC.

**Columns (Explicit Mapping):**
- `date`: The month of the record. if you need to aggregate by month or year, use the date as it also contains information about the month and year.
- `entity`: The name of the generating unit.
- `tariff_gel`: The regulated tariff in GEL/MWh.
- `tariff_usd`: The regulated tariff in USD/MWh.

**Crucial Logic:**
- **For thermal plants:** A missing tariff in a given month means the plant did not generate electricity.
- **For hydro plants:** A missing tariff from a given month onwards means the plant was deregulated from that point on.

---
### Table: public.tech_quantity ###
Covers total electricity demand and supply side quantities.

**Columns (Explicit Mapping):**
- `date`: The month of the record. if you need to aggregate by month or year, use the date as it also contains information about the month and year.
- `type_tech`: The category of supply, demand, or transit.
- `quantity_tech`: The amount of electricity in **thousand MWh**.

**Key Synonyms for `type_tech` column:**
- User says "hydro generation", "hydro power", "HPP" -> Query for `type_tech = 'hydro'`
- User says "thermal generation", "thermal power", "TPP" -> Query for `type_tech = 'thermal'`
- User says "wind generation", "wind power plant", "wind turbine" -> Query for `type_tech = 'wind'`
- User says "solar generation" -> There is no 'solar' type in this table; check other tables.
- User says "exports" -> Query for `type_tech = 'export'`
- User says "imports" -> Query for `type_tech = 'import'`
- User says "losses" -> Query for `type_tech = 'losses'`
- User says "Abkhazeti consumption" -> Query for `type_tech = 'abkhazeti'`
- User says "transit" -> Query for `type_tech = 'transit'` (Note: Transit increased in 2022-2023 due to high gas prices in Turkey).

---
### Table: public.trade ###
Shows electricity trade outcomes broken down by entity and market segment.

**Columns (Explicit Mapping):**
- `date`: The month of the record. if you need to aggregate by month or year, use the date as it also contains information about the month and year.
- `entity`: The source of electricity selling it.
- `segment`: The market segment.
- `quantity`: The traded energy amount in **thousand MWh**.

**Important Notes:**
- Bilateral and exchange trades are currently summed together.
- `thermal_ppa` and `renewable_ppa` entities must sell all their generation to ESCO during mandatory periods and cannot trade bilaterally.

### Data Granularity & Timeframes ###
- **Granularity:** All tables with a `date` column contain **monthly** data. The date typically represents the first day of the month. The data is NOT available at a daily or weekly level.
"""
