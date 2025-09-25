import os
import re
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_core.agents import AgentAction
from langchain_core.tools import Tool  # Import the base Tool class

from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Assumes context.py contains your DB_SCHEMA_DOC string
from context import DB_SCHEMA_DOC

# ---------------- (Code from previous version is unchanged until the /ask endpoint) ----------------
# Logging, Env, Database, FastAPI, System Prompt, Pydantic Models,
# and all helper/analytics functions remain the same as the last version.
# For brevity, I am skipping to the corrected API endpoint.
# ... (All previous helper code is assumed to be here) ...

# ---------------- API Endpoint with the FIX ----------------

@app.post("/ask", response_model=APIResponse)
def ask(q: Question, x_app_key: str = Header(...)):
    start_time = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY, request_timeout=60)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        all_tools = toolkit.get_tools()

        # === THE FIX IS HERE ===
        # Instead of modifying the tool in-place, we create a new one that wraps the original.
        
        # 1. Find the original SQL query tool provided by the toolkit.
        original_sql_tool = next((tool for tool in all_tools if tool.name == "sql_db_query"), None)
        
        final_tools = all_tools
        if original_sql_tool:
            # 2. Define a new function that wraps the original tool's execution.
            def wrapped_sql_run_func(query: str):
                logger.info(f"WRAPPED TOOL: Intercepted query: {query}")
                # Apply our sanitization and validation logic first.
                cleaned_query = clean_sql(query)
                validate_sql_is_safe(cleaned_query)
                
                # Now, call the original tool's .run() method with the cleaned query.
                return original_sql_tool.run(cleaned_query)

            # 3. Create a new Tool with the same name and description, but using our wrapped function.
            wrapped_tool = Tool(
                name="sql_db_query",
                func=wrapped_sql_run_func,
                description=original_sql_tool.description + " NOTE: This tool is wrapped. LIMIT clauses are automatically removed for full analysis."
            )
            
            # 4. Replace the original tool in the list with our new wrapped version.
            final_tools = [wrapped_tool if tool.name == "sql_db_query" else tool for tool in all_tools]
        # === END OF FIX ===

        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            tools=final_tools, # Use the new list with the wrapped tool
            verbose=True,
            agent_type="openai-tools",
            system_message=SYSTEM_PROMPT,
            max_iterations=8,
            early_stopping_method="generate",
        )

        # The rest of the logic remains the same
        result = agent.invoke({"input": q.query}, return_intermediate_steps=True)
        sql = extract_sql_from_steps(result.get("intermediate_steps", []))

        if not sql:
            logger.warning("No SQL extracted, falling back to agent's direct output.")
            final_answer = scrub_schema_mentions(result.get("output", "I could not determine how to answer that question."))
        else:
            # Your process_and_analyze_data function will now receive SQL that has already been cleaned
            final_answer = process_and_analyze_data(sql, q.query, llm)
        
        return APIResponse(answer=final_answer, execution_time=round(time.time() - start_time, 2))

    except SQLAlchemyError as e:
        logger.error(f"DB error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error")
    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Processing error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")
