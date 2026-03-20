from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from pydantic import BaseModel, Field
from pathlib import Path
from shutil import copyfileobj

class Response:
    def __init__(self, code=200, message='success', data=None):
        self.code=code
        self.message = message
        self.data= data

from AgentTools import (
    ALL_TOOLS,
    DATA_AGENT_SYSTEM_PROMPT,
)

agent = create_agent(
    model=ChatTongyi(model='qwen3-max'),
    tools=ALL_TOOLS,
    system_prompt=DATA_AGENT_SYSTEM_PROMPT
)


def load_file(file,user_id):
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{user_id}.csv"

    if hasattr(file, "file"):
        file.file.seek(0)
        with path.open("wb") as f:
            copyfileobj(file.file, f)
    elif isinstance(file, (bytes, bytearray)):
        path.write_bytes(file)
    elif isinstance(file, str):
        with open(file, "rb") as src, path.open("wb") as dst:
            copyfileobj(src, dst)
    else:
        raise ValueError("不支持的文件类型，请传入 UploadFile、bytes 或文件路径")

    return f"/data/{user_id}.csv"


def process(file_path,user_prompt):
    agent = create_agent(
        model=ChatTongyi(model='qwen3-max'),
        tools=ALL_TOOLS,
        system_prompt=DATA_AGENT_SYSTEM_PROMPT
    )
    report = agent.invoke({
        "messages": [
            {"role": "user", "content": user_prompt},
        ]
    })
    return report
