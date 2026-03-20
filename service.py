from pathlib import Path
from shutil import copyfileobj

from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi

from AgentTools import ALL_TOOLS, DATA_AGENT_SYSTEM_PROMPT


agent = create_agent(
    model=ChatTongyi(model="qwen3-max"),
    tools=ALL_TOOLS,
    system_prompt=DATA_AGENT_SYSTEM_PROMPT,
)


def load_file(file, user_id):
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
        raise ValueError("???????????? UploadFile?bytes ?????")

    return f"/data/{user_id}.csv"


def _resolve_path(file_path: str) -> Path:
    if file_path.startswith("/data/"):
        return Path(__file__).resolve().parent / file_path.lstrip("/")
    return Path(file_path).resolve()


def process(file_path, user_prompt):
    local_path = _resolve_path(file_path)
    if not local_path.exists():
        raise FileNotFoundError(f"?????: {local_path}")

    task_prompt = (
        f"?????CSV????????: {local_path}\n"
        f"????: {user_prompt}\n"
        "????????????????????????????Markdown???"
    )

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": task_prompt},
            ]
        }
    )

    report = ""
    if isinstance(result, dict):
        messages = result.get("messages") or []
        if messages:
            last = messages[-1]
            report = getattr(last, "content", "") or str(last)
        if not report:
            report = str(result)
    else:
        report = str(result)

    return {
        "file_path": str(local_path),
        "report": report,
        "raw": str(result),
    }
