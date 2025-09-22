DB_SCHEMA_DOC = """
General Rules for the Assistant:
• You are NOT allowed to return or export full tables, rows, or detailed datasets from the database.
• You may only provide summaries, insights, comparisons, and explanations based on the data.
• If a user asks for raw data dumps, you must refuse and instead suggest insights or aggregated views.

General Remarks:
• Volume in “Table: public.energy_balance_long” is measured in TJ.
• Quantities in all other tables/views are in “thousand MWH”
• Prices are either in GEL/MWH, or USD/MWH

Table: public.energy_balance_long
The table show the energy consumption of the different sectors. Yearly data is based on the National Energy Balances provided by the National Statistics Office – GEOSTAT.
• year: calendar year.
• sector: The sectors which consume energy -  
  A) Industry (or industry sector) which include: Iron and steel; Chemical (including petrochemical); Non-metallic minerals; Mining and quarrying; Food, beverages and tobacco; Construction; Other Industry  
  B) Transport (or transport sector), which include: Road; Rail; Pipeline transport; Other Transport.  
  C) Other use, which include: Commercial and public services; Residential; Other.
• energy_source: different energy sources consumed in Georgia. There are specific sources (Coal, Crude Oil, Oil products, Natural Gas, Nuclear, Hydro, Geothermal, wind, solar etc; Biofuel & Waste; Electricity; Heat) and “Total” is a sum of all energy sources. 
• volume_tj: energy consumption measured in TJ.

Table: public.entities
This table just show the entities for which the information is shown in other view and tables. Cover the entities in electricity sector.
• entity: name of the entity.
• entity_normalized: normalized unique identifier. Some entities are called different names. For analyses, always use entity_normalized instead of entity because names may vary.
• type: entity classification (HPP, TPP, Import, Export, Solar, Wind).
• ownership: owner of the entity.
• source: shows whether the entity is local – locally produced hydro or wind, or it is import_dependent – directly import or gas import dependent thermal power plants.

Table: public.monthly_cpi
The consumer price index (CPI) is calculated by the National Statistics Office – GEOSTAT. It shows CPI indexed to 2015 = 100 (base year). The aim is to compare energy CPI with Overall CPI and CPI of some specific goods. In period dec2020-feb2021 we see big drop in electricity_gas_and_other_fuels CPI, due to energy subsidies provided to Georgian households.
• date: monthly timestamp
• cpi_type: overall CPI, electricity_gas_and_other_fuels and other specific goods.
• cpi: consumer price index (numeric).

Table: public.price
• date: monthly timestamp
• p_dereg_gel – price (GEL/MWH) deregulated power plants (all hydro at the moment) receive when selling as balancing electricity (unsold bilaterally or on exchange from July 2024).
• p_dereg_usd – the same as p_dereg_gel but in USD/MWH (derived by dividing p_dereg_gel by xrate).
• p_bal_gel – Balancing electricity price (GEL/MWH). Weighted average of all balancing sales. PPAs (thermal_ppa, renewable_ppa from `public.trade`) must sell as balancing during mandatory purchase periods (8–12 months). In practice, usually the most expensive sources sell here, though occasionally cheap regulated/deregulated HPPs appear (esp. Apr–Jun).
• p_bal_usd – the same as p_bal_gel but in USD/MWH (derived as p_bal_gel / xrate).
• p_gcap_gel – guaranteed capacity fee/charge (GEL/MWH). Paid by all final consumers and exporters to cover fixed costs of guaranteed capacity TPPs. Not paid by Abkhazia (occupied territory). Calculated as fixed monthly fee / (total consumption – Abkhazia – exports).
• p_gcap_usd – same as p_gcap_gel but in USD/MWH (derived as p_gcap_gel / xrate).
• xrate – Exchange rate GEL/USD.

Table: public.tariff_gen
The table show the tariffs of the power plant approved by the national energy regulator – GNERC.  
For thermals: if tariff is missing in a month, it means no generation → no tariff (variable cost only, fixed covered by gcap).  
For hydros: if tariff is missing from a given month, the HPP was deregulated and no longer subject to tariff approval.
• date: monthly timestamp
• entity: generating unit.
• tariff_gel: regulated tariff in GEL/MWh.
• tariff_usd: regulated tariff in USD/MWh.

Table: public.tech_quantity
Covers the demand and supply sides.
Demand side:
• abkhazeti: "Abkhazeti"
• supply_distribution: "Supplier/Distributor"
• direct_customers: "Direct Consumers"
• self_cons: "Self-Consumption by PP"
• losses: "Losses"
• export: "Export"
Supply side:
• hydro: "Hydro Generation"
• thermal: "Thermal Generation"
• wind: "Wind Generation"
• import: "Import"
Transit:
• transit: electricity transiting (Russia/Azerbaijan → Turkey). Increased 2022–2023 due to high gas prices in Turkey (Russia–Ukraine crisis).
Columns:
• date: monthly timestamp
• type_tech: demand, supply, or transit category
• quantity_tech: electricity quantity (1000 MWh).

Table: public.trade
Shows electricity trade outcomes.  
Unlike `tech_quantity`, which shows totals, this is broken down by trade.  
Electricity may be sold bilaterally or on the exchange (introduced July 2024). Any remaining goes to balancing.  
⚠ Note: bilateral and exchange trades are not separated in this data — only the sum is shown.  
Thermal_ppa and renewable_ppa cannot sell bilaterally or on exchange during mandatory sale period; all their generation goes to ESCO. Outside this period, PPAs may sell across all markets.
• date: monthly timestamp
• entity: the source (plant, import, or aggregation).
• segment: market segment (day-ahead, balancing, bilateral, etc.).
• quantity: traded energy (1000 MWh).
"""
