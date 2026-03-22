import json
import logging
from pathlib import Path
from shutil import copy2, copyfileobj

from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
import pandas as pd

from AgentTools import ALL_TOOLS, DATA_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


agent = create_agent(
    model=ChatTongyi(model="qwen-max"),
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
        raise ValueError("不支持的文件类型，请传入 UploadFile、bytes 或文件路径字符串")

    return f"/data/{user_id}.csv"


def _resolve_path(file_path: str) -> Path:
    if file_path.startswith("/data/"):
        return Path(__file__).resolve().parent / file_path.lstrip("/")
    return Path(file_path).resolve()


def _json_payload(content):
    if not isinstance(content, str):
        return {}
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize_path(path_value):
    if not path_value:
        return None
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()


def _extract_tool_paths(messages):
    report_path = None
    save_path = None

    for msg in messages:
        name = getattr(msg, "name", None)
        if not name:
            continue
        payload = _json_payload(getattr(msg, "content", ""))
        output_path = _normalize_path(payload.get("output_path"))
        if not output_path:
            continue
        if name == "export_report":
            report_path = output_path
        elif name == "save_dataset":
            save_path = output_path

    return report_path, save_path


def _extract_report_from_messages(messages):
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and "# 数据处理报告" in content:
            return content
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _read_text_file(path: Path):
    if not path or not path.exists():
        return ""
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue
    return ""


def _build_preview(path: Path, n: int = 10):
    if not path.exists():
        return {"columns": [], "rows": []}

    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(path, nrows=max(1, n), encoding=encoding)
            df = df.where(pd.notna(df), None)
            return {
                "columns": [str(c) for c in df.columns],
                "rows": df.to_dict(orient="records"),
            }
        except Exception:
            continue

    return {"columns": [], "rows": []}


def process(file_path, user_prompt):
    local_path = _resolve_path(file_path)
    if not local_path.exists():
        raise FileNotFoundError(f"文件不存在: {local_path}")

    fixed_report_path = local_path.with_suffix(".md")
    if fixed_report_path.exists():
        try:
            fixed_report_path.unlink()
        except Exception:
            logger.warning("删除旧报告失败: %s", fixed_report_path)

    task_prompt = (
        f"请处理这个 CSV 文件，文件路径：{local_path}\n"
        f"用户需求：{user_prompt}\n"
        "请严格按照系统提示完成数据处理，并输出最终 Markdown 报告。"
    )

    try:
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": task_prompt},
                ]
            }
        )
    except Exception as exc:
        logger.exception("Agent 调用失败")
        raise RuntimeError(f"Agent 调用失败: {exc}") from exc

    messages = []
    if isinstance(result, dict):
        messages = result.get("messages") or []

    report_path, save_path = _extract_tool_paths(messages)

    # Keep processed CSV path fixed (e.g., 6.csv).
    if save_path and save_path.exists() and save_path.resolve() != local_path.resolve():
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            copy2(save_path, local_path)
        except Exception as exc:
            logger.warning("覆盖输出 CSV 失败: %s", exc)

    # Keep report path fixed with same basename (e.g., 6.md).
    if report_path and report_path.exists() and report_path.resolve() != fixed_report_path.resolve():
        try:
            fixed_report_path.parent.mkdir(parents=True, exist_ok=True)
            copy2(report_path, fixed_report_path)
        except Exception as exc:
            logger.warning("同步报告文件失败: %s", exc)

    report = _read_text_file(fixed_report_path)
    if not report and report_path:
        report = _read_text_file(report_path)
    if not report:
        report = _extract_report_from_messages(messages)
    if not report:
        report = str(result)

    preview = _build_preview(local_path, n=10)

    return {
        "file_path": str(local_path),
        "report": report,
        "raw": str(result),
        "report_path": str(fixed_report_path),
        "preview": preview,
    }
