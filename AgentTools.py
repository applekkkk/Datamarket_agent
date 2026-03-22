from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd
from langchain_core.tools import tool

class FrameRegistry:
    def __init__(self) -> None:
        self._frames: dict[str, pd.DataFrame] = {}
        self._sources: dict[str, str] = {}
        self._op_log: list[dict[str, Any]] = []

    def put(self, frame_id: str, df: pd.DataFrame, source_path: str | None = None) -> None:
        self._frames[frame_id] = df
        if source_path:
            self._sources[frame_id] = source_path

    def get(self, frame_id: str) -> pd.DataFrame:
        if frame_id not in self._frames:
            raise KeyError(f"frame '{frame_id}' not found")
        return self._frames[frame_id]

    def get_source(self, frame_id: str) -> str:
        return self._sources.get(frame_id, "")

    def list_frame_ids(self) -> list[str]:
        return list(self._frames.keys())

    def drop(self, frame_id: str) -> None:
        self._frames.pop(frame_id, None)
        self._sources.pop(frame_id, None)

    def log(self, tool_name: str, params: dict[str, Any], summary: str) -> None:
        self._op_log.append(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "tool": tool_name,
                "params": params,
                "summary": summary,
            }
        )

    def get_log(self) -> list[dict[str, Any]]:
        return list(self._op_log)

    def clear(self) -> int:
        n = len(self._frames)
        self._frames.clear()
        self._sources.clear()
        self._op_log.clear()
        return n


_reg = FrameRegistry()


def _safe_frame_id() -> str:
    return f"frame_{datetime.now().strftime('%H%M%S%f')[:10]}"


@tool(description="加载本地CSV到内存，返回frame_id。")
def load_dataset(path: str, encoding: str = "utf-8", sep: str = ",") -> dict:
    df = pd.read_csv(path, encoding=encoding, sep=sep)
    frame_id = _safe_frame_id()
    _reg.put(frame_id, df, source_path=path)
    _reg.log("load_dataset", {"path": path}, f"loaded {len(df)}x{len(df.columns)}")
    return {
        "frame_id": frame_id,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "source_path": path,
    }


@tool(description="列出当前内存里的frame。")
def list_frames() -> list[dict]:
    return [
        {
            "frame_id": fid,
            "rows": int(_reg.get(fid).shape[0]),
            "cols": int(_reg.get(fid).shape[1]),
            "source_path": _reg.get_source(fid),
        }
        for fid in _reg.list_frame_ids()
    ]


@tool(description="获取字段类型、空值统计。")
def get_dataset_info(frame_id: str) -> dict:
    df = _reg.get(frame_id)
    columns = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        columns.append(
            {
                "name": str(col),
                "dtype": str(df[col].dtype),
                "null_count": null_count,
                "null_pct": round(null_count / len(df) * 100, 2) if len(df) else 0,
            }
        )
    return {
        "frame_id": frame_id,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": columns,
    }


@tool(description="获取前N行预览。")
def get_preview(frame_id: str, n: int = 10) -> dict:
    df = _reg.get(frame_id)
    sample = df.head(max(1, n)).where(pd.notna(df.head(max(1, n))), other=None)
    return {
        "frame_id": frame_id,
        "columns": [str(c) for c in sample.columns],
        "rows": sample.to_dict(orient="records"),
    }


@tool(description="用DuckDB执行SQL。SQL中直接用 frame_id 作为表名。")
def run_sql(frame_id: str, sql: str, output_frame_id: str = "") -> str | dict[str, int | str | list[str]]:
    df = _reg.get(frame_id)
    query = str(sql or "").strip().rstrip(";")
    if not query:
        raise ValueError("SQL is empty")

    con = duckdb.connect(database=":memory:")
    try:
        con.register(frame_id, df)          # ← 表名改成 frame_id，不用 input_table
        result_df = con.execute(query).fetchdf()
    except Exception as exc:
        cols = {str(c): str(df[c].dtype) for c in df.columns}
        return (
            f"SQL 执行失败：{exc}\n"
            f"出错 SQL：{query}\n"
            f"表名应为：{frame_id}\n"
            f"可用字段及类型：{cols}"
        )
    finally:
        con.close()

    target = (output_frame_id or frame_id).strip() or frame_id
    _reg.put(target, result_df, source_path=_reg.get_source(frame_id))
    _reg.log("run_sql", {"frame_id": frame_id, "output_frame_id": target, "sql": query},
             f"result {len(result_df)}x{len(result_df.columns)}")
    return {
        "frame_id": target,
        "rows": int(result_df.shape[0]),
        "cols": int(result_df.shape[1]),
        "columns": [str(c) for c in result_df.columns],
    }
@tool(description="保存frame到文件，支持csv/json/parquet。")
def save_dataset(frame_id: str, output_path: str, format: str = "csv", index: bool = False) -> dict:
    df = _reg.get(frame_id)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt = (format or "csv").lower()
    if fmt == "csv":
        df.to_csv(out, index=index)
    elif fmt == "json":
        df.to_json(out, orient="records", force_ascii=False, indent=2)
    elif fmt == "parquet":
        df.to_parquet(out, index=index)
    else:
        raise ValueError("format must be one of: csv/json/parquet")

    _reg.log("save_dataset", {"frame_id": frame_id, "output_path": str(out), "format": fmt}, "saved")
    return {"output_path": str(out), "format": fmt, "rows": int(df.shape[0]), "cols": int(df.shape[1])}


@tool(description="导出Markdown处理报告。")
def export_report(frame_id: str, output_path: str, user_goal: str = "") -> dict:
    df = _reg.get(frame_id)
    log = _reg.get_log()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    goal_text = user_goal if user_goal else "未提供"

    lines = [
        "# 数据处理报告",
        "",
        "## 一、任务概览",
        f"- 用户目标：{goal_text}",
        f"- 执行时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 二、处理过程（按时间顺序）",
    ]

    if log:
        for i, item in enumerate(log, 1):
            lines.append(f"{i}. **{item['tool']}** ({item['time']})")
            lines.append(f"   - 说明：{item['summary']}")
            lines.append(f"   - 参数：`{json.dumps(item['params'], ensure_ascii=False)}`")
    else:
        lines.append("- 无操作日志")

    lines.extend(
        [
            "",
            "## 三、结果概览",
            f"- 行数：{int(df.shape[0])}",
            f"- 列数：{int(df.shape[1])}",
            f"- 字段：{', '.join([str(c) for c in df.columns])}",
        ]
    )

    out.write_text("\n".join(lines), encoding="utf-8")
    _reg.log("export_report", {"frame_id": frame_id, "output_path": str(out)}, "report exported")
    return {"output_path": str(out), "steps_count": len(log)}


@tool(description="查看最近N条操作日志。")
def get_operation_log(last_n: int = 0) -> list[dict]:
    log = _reg.get_log()
    if last_n and last_n > 0:
        return log[-last_n:]
    return log


@tool(description="释放指定frame。")
def drop_frame(frame_id: str) -> dict:
    _reg.drop(frame_id)
    return {"dropped": frame_id, "remaining_frames": _reg.list_frame_ids()}


@tool(description="清空全部frame和日志。")
def clear_all_frames() -> dict:
    count = _reg.clear()
    return {"cleared_count": count}


ALL_TOOLS = [
    load_dataset,
    list_frames,
    get_dataset_info,
    get_preview,
    run_sql,
    save_dataset,
    export_report,
    get_operation_log,
    drop_frame,
    clear_all_frames,
]


DATA_AGENT_SYSTEM_PROMPT = """
你是一个面向结构化表格数据的智能处理助手。

规则：
1）先调用 load_dataset，记录它返回的 frame_id。
2）调用 get_dataset_info 和 get_preview，传入上一步返回的 frame_id。
3）所有处理通过 run_sql 完成，SQL 中的表名必须是 load_dataset 实际返回的 frame_id。
4）中文字段名用双引号包裹，例如："委托价格"。
5）每步 run_sql 后说明影响行列。
6）完成后调用 save_dataset 覆盖原文件，再调用 export_report 生成同名 md。
7）SQL 报错时先调用 get_dataset_info 确认字段名再重试。

重要：frame_id 是动态生成的，每次运行都不同，必须从 load_dataset 的返回值中读取，
禁止自行编造或使用任何固定字符串作为 frame_id。
""".strip()