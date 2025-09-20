DB_SCHEMA_DOC = """
General Remarks:
• Volume in “Table: public.energy_balance_long” is measured in TJ.
• Quantities in all other tables/views are in “thousand MWH”
• Prices are either in GEL/MWH, or USD/MWH

Table: public.energy_balance_long
The table shows the energy consumption of the different sectors. Yearly data is based on the National Energy Balances provided by the National Statistics Office – GEOSTAT.
• year: calendar year.
• sector: The sectors which consume energy -
  A) Industry sector: Iron and steel; Chemical (including petrochemical); Non-metallic minerals; Mining and quarrying; Food, beverages and tobacco; Construction; Other Industry
  B) Transport sector: Road; Rail; Pipeline transport; Other Transport
  C) Other use: Commercial and public services; Residential; Other
• energy_source: sources consumed in Georgia (Coal, Crude Oil, Oil products, Natural Gas, Nuclear, Hydro, Geothermal, Wind, Solar, Biofuel & Waste, Electricity, Heat). “Total” is the sum of all.
• volume_tj: energy consumption measured in TJ.

Table: public.entities
Covers the entities in the electricity sector.
• entity: name of the entity.
• entity_normalized: normalized unique identifier. Use entity_normalized instead of entity for analyses because names may vary.
• type: classification (HPP, TPP, Import, Export, Solar, Wind).
• ownership: ownership type.
• source: local (hydro, wind) or import_dependent (import or gas-fuel TPPs).

MV: public.monthly_cpi_mv
Consumer price index (CPI) from GEOSTAT, base year 2015 = 100.
• date: monthly timestamp
• cpi_type: overall CPI, electricity_gas_and_other_fuels, or other goods.
• cpi: index value. Electricity CPI dropped in Dec2020–Feb2021 due to household subsidies.

MV: public.price_with_usd
• date: monthly timestamp
• p_dereg_gel/usd: price (GEL/MWh or USD/MWh) paid to deregulated HPPs for balancing electricity.
• p_bal_gel/usd: balancing electricity price (weighted average).
• p_gcap_gel/usd: guaranteed capacity fee (covers fixed costs of thermal plants). Not paid by Abkhazia.
• xrate: exchange rate GEL/USD.

MV: public.tariff_with_usd
Regulated tariffs set by GNERC.
• date: monthly timestamp
• entity: generating unit.
• tariff_gel: GEL/MWh
• tariff_usd: USD/MWh
Missing tariff for HPP → deregulated. Missing for TPP → no generation that month.

MV: public.tech_quantity_view
Covers demand and supply side.
Demand: abkhazeti, supply_distribution, direct_customers, self_cons, losses, export.
Supply: hydro, thermal, wind, import.
Transit: electricity flows from Russia/Azerbaijan to Turkey. Increased 2022–2023 due to high gas prices in Turkey.
• date: monthly timestamp
• type_tech: indicator
• quantity_tech: electricity quantity (1000 MWh).

Table: public.trade
Trade outcomes: bilateral, exchange (introduced July 2024), balancing.
Note: no split between bilateral and exchange in data, only combined.
PPAs (thermal_ppa, renewable_ppa) must sell via ESCO during mandatory purchase period (8–12 months).
• date: monthly timestamp
• entity: source (plant, import, or aggregate)
• segment: market segment
• quantity: traded energy volume (1000 MWh).

MV: public.trade_derived_entities
Aggregates trades into standardized categories.
• date
• entity: derived category
• segment
• quantity
Categories: regulated_hpp, regulated_old_tpp, renewable_ppa, deregulated_hydro, regulated_new_tpp, thermal_ppa, import, transit, total_hpp, total_thermal.

MV: public.trade_by_type
Aggregates traded electricity by entity type (segment = 'total').
• date
• type: HPP, TPP, WPP, Import, Export, Transit
• quantity
Usage: compare type-level contributions, seasonal/yearly analysis.

MV: public.trade_by_source
Aggregates electricity by source (segment = 'total').
• date
• source: local vs import_dependent
• quantity
Usage: measure energy security and import-dependency.

MV: public.trade_by_ownership
Aggregates electricity by ownership (segment = 'total').
• date
• ownership: state-owned, private, PPA, foreign-owned
• quantity
Usage: analyze market shares, privatization trends, and security of supply.
"""
