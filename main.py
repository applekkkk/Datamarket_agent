from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from pydantic import BaseModel, Field

from AgentTools import (
    AGENT_TOOLS,
    DATA_AGENT_SYSTEM_PROMPT,
    add_tier_column,
    clean_missing_values,
    convert_currency_column,
    drop_frame,
    get_dataset_info,
    get_preview,
    load_dataset,
    save_dataset,
)

app = FastAPI(title="Data Agent Service")

@app.post("/agent/process")
async def process(prompt):
    return

@app.post("/download")
async def download():
    return
