"""Data processing tools for an AI Agent."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

FRAME_STORE: Dict[str, pd.DataFrame] = {}


def _parse_columns(columns: Optional[str | List[str]]) -> Optional[List[str]]:
    if columns is None:
        return None
    if isinstance(columns, list):
        return [str(c).strip() for c in columns if str(c).strip()]
    return [c.strip() for c in str(columns).split(",") if c.strip()]


def _require_frame(frame_id: str) -> pd.DataFrame:
    if frame_id not in FRAME_STORE:
        raise ValueError(f"Unknown frame_id: {frame_id}")
    return FRAME_STORE[frame_id]


def _safe_preview(df: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    return df.head(max(1, int(limit))).replace({np.nan: None}).to_dict(orient="records")


@tool(description="查看当前内存里有哪些数据表（用于调试和流程编排）")
def list_frames() -> Dict[str, Any]:
    items = [
        {"frame_id": fid, "rows": int(df.shape[0]), "cols": int(df.shape[1])}
        for fid, df in FRAME_STORE.items()
    ]
    return {"ok": True, "count": len(items), "frames": items}


@tool(description="释放指定数据表，避免内存持续增长")
def drop_frame(frame_id: str) -> Dict[str, Any]:
    existed = frame_id in FRAME_STORE
    FRAME_STORE.pop(frame_id, None)
    return {"ok": True, "removed": bool(existed), "frame_id": frame_id}


@tool(description="读取 CSV 或 Excel 文件到内存并返回 frame_id")
def load_dataset(
    file_path: str,
    frame_id: Optional[str] = None,
    file_type: str = "auto",
    encoding: str = "utf-8-sig",
) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    mode = file_type.lower().strip() if file_type else "auto"
    if mode == "auto":
        mode = "excel" if ext in {".xls", ".xlsx"} else "csv"

    if mode == "excel":
        df = pd.read_excel(file_path)
    elif mode == "csv":
        df = pd.read_csv(file_path, encoding=encoding)
    else:
        raise ValueError("file_type must be one of: auto, csv, excel")

    fid = frame_id or f"df_{uuid.uuid4().hex[:8]}"
    FRAME_STORE[fid] = df
    return {
        "ok": True,
        "frame_id": fid,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns.tolist()],
        "preview": _safe_preview(df, limit=5),
    }


@tool(description="获取数据表字段画像与质量概要（类型、空值、唯一值、统计）")
def get_dataset_info(frame_id: str, max_unique: int = 20) -> Dict[str, Any]:
    df = _require_frame(frame_id)
    summary: List[Dict[str, Any]] = []

    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        unique_count = int(s.nunique(dropna=True))
        info: Dict[str, Any] = {
            "column": str(col),
            "dtype": str(s.dtype),
            "non_null": non_null,
            "null_count": int(s.isna().sum()),
            "unique_count": unique_count,
        }

        if unique_count <= max_unique:
            vals = s.dropna().unique().tolist()
            info["sample_values"] = [str(v) for v in vals[:max_unique]]

        if pd.api.types.is_numeric_dtype(s):
            info["min"] = None if s.dropna().empty else float(s.min())
            info["max"] = None if s.dropna().empty else float(s.max())
            info["mean"] = None if s.dropna().empty else float(s.mean())

        summary.append(info)

    return {
        "ok": True,
        "frame_id": frame_id,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "summary": summary,
    }


@tool(description="查看数据表前 N 行预览")
def get_preview(frame_id: str, limit: int = 10) -> Dict[str, Any]:
    df = _require_frame(frame_id)
    return {
        "ok": True,
        "frame_id": frame_id,
        "rows": int(df.shape[0]),
        "preview": _safe_preview(df, limit=limit),
    }


@tool(description="缺失值处理：支持删行、删列或填充")
def clean_missing_values(
    frame_id: str,
    strategy: str = "drop_rows",
    columns: Optional[str | List[str]] = None,
    fill_value: Optional[str] = None,
) -> Dict[str, Any]:
    df = _require_frame(frame_id)
    cols = _parse_columns(columns)

    before_rows, before_cols = int(df.shape[0]), int(df.shape[1])
    nulls_before = int(df.isna().sum().sum())

    if strategy == "drop_rows":
        result = df.dropna(subset=cols) if cols else df.dropna()
    elif strategy == "drop_columns":
        result = df.dropna(axis=1) if cols is None else df.drop(columns=[c for c in cols if c in df.columns])
    elif strategy == "fill":
        if cols:
            result = df.copy()
            for c in cols:
                if c in result.columns:
                    result[c] = result[c].fillna(fill_value)
        else:
            result = df.fillna(fill_value)
    else:
        raise ValueError("strategy must be one of: drop_rows, drop_columns, fill")

    FRAME_STORE[frame_id] = result
    nulls_after = int(result.isna().sum().sum())
    return {
        "ok": True,
        "frame_id": frame_id,
        "before": {"rows": before_rows, "cols": before_cols, "null_count": nulls_before},
        "after": {
            "rows": int(result.shape[0]),
            "cols": int(result.shape[1]),
            "null_count": nulls_after,
        },
        "affected_rows": before_rows - int(result.shape[0]),
    }


@tool(description="按汇率转换货币列并生成新列")
def convert_currency_column(
    frame_id: str,
    source_column: str,
    target_column: str = "price_cny",
    rate: float = 7.24,
    round_digits: int = 2,
) -> Dict[str, Any]:
    df = _require_frame(frame_id).copy()
    if source_column not in df.columns:
        raise ValueError(f"Column not found: {source_column}")

    numeric = pd.to_numeric(df[source_column], errors="coerce")
    df[target_column] = (numeric * float(rate)).round(int(round_digits))
    FRAME_STORE[frame_id] = df

    return {
        "ok": True,
        "frame_id": frame_id,
        "source_column": source_column,
        "target_column": target_column,
        "rate": float(rate),
        "converted_non_null": int(df[target_column].notna().sum()),
    }


@tool(description="基于阈值生成分层标签列（bronze/silver/gold）")
def add_tier_column(
    frame_id: str,
    amount_column: str,
    target_column: str = "tier",
    bronze_max: float = 100,
    silver_max: float = 500,
) -> Dict[str, Any]:
    df = _require_frame(frame_id).copy()
    if amount_column not in df.columns:
        raise ValueError(f"Column not found: {amount_column}")

    amount = pd.to_numeric(df[amount_column], errors="coerce")
    conditions = [amount < bronze_max, (amount >= bronze_max) & (amount <= silver_max), amount > silver_max]
    choices = ["bronze", "silver", "gold"]
    df[target_column] = np.select(conditions, choices, default=np.nan)

    FRAME_STORE[frame_id] = df
    counts = df[target_column].value_counts(dropna=False).to_dict()
    return {
        "ok": True,
        "frame_id": frame_id,
        "target_column": target_column,
        "thresholds": {"bronze_max": bronze_max, "silver_max": silver_max},
        "distribution": {str(k): int(v) for k, v in counts.items()},
    }


@tool(description="按条件过滤行：支持 query 表达式与指定列非空")
def filter_rows(
    frame_id: str,
    query_expr: Optional[str] = None,
    required_not_null: Optional[str | List[str]] = None,
) -> Dict[str, Any]:
    df = _require_frame(frame_id)
    result = df

    if required_not_null:
        cols = _parse_columns(required_not_null) or []
        existing = [c for c in cols if c in result.columns]
        if existing:
            result = result[result[existing].notna().all(axis=1)]

    if query_expr:
        result = result.query(query_expr)

    before = int(df.shape[0])
    after = int(result.shape[0])
    FRAME_STORE[frame_id] = result.copy()

    return {
        "ok": True,
        "frame_id": frame_id,
        "before_rows": before,
        "after_rows": after,
        "removed_rows": before - after,
    }


@tool(description="用 IQR 规则检测异常值，可选写回标记列")
def detect_outliers_iqr(
    frame_id: str,
    column: str,
    whisker: float = 1.5,
    mark_column: Optional[str] = None,
) -> Dict[str, Any]:
    df = _require_frame(frame_id).copy()
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        return {"ok": True, "frame_id": frame_id, "column": column, "outlier_count": 0, "indexes": []}

    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - whisker * iqr, q3 + whisker * iqr

    mask = pd.to_numeric(df[column], errors="coerce").between(lower, upper, inclusive="both")
    outlier_mask = ~mask & df[column].notna()
    indexes = df.index[outlier_mask].tolist()

    if mark_column:
        df[mark_column] = outlier_mask
        FRAME_STORE[frame_id] = df

    return {
        "ok": True,
        "frame_id": frame_id,
        "column": column,
        "lower": float(lower),
        "upper": float(upper),
        "outlier_count": int(len(indexes)),
        "indexes": [int(i) for i in indexes[:200]],
    }


@tool(description="保存数据表到 CSV 或 Excel 文件")
def save_dataset(frame_id: str, output_path: str, index: bool = False) -> Dict[str, Any]:
    df = _require_frame(frame_id)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=index)
    else:
        df.to_csv(output_path, index=index, encoding="utf-8-sig")

    return {
        "ok": True,
        "frame_id": frame_id,
        "output_path": output_path,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }


AGENT_TOOLS = [
    list_frames,
    drop_frame,
    load_dataset,
    get_dataset_info,
    get_preview,
    clean_missing_values,
    convert_currency_column,
    add_tier_column,
    filter_rows,
    detect_outliers_iqr,
    save_dataset,
]

DATA_AGENT_SYSTEM_PROMPT = """
你是一个面向结构化表格数据的数据处理助手，目标是产出可复现、可追踪的处理结果。

数据生命周期规则：
1）加载阶段：先用 load_dataset 读取数据，并记录 frame_id。
2）分析阶段：处理前必须先调用 get_dataset_info / get_preview 了解字段与数据质量。
3）处理阶段：每次只做一个明确变换步骤，并说明变更原因。
4）校验阶段：每次变换后都复核行数、空值、关键字段分布。
5）保存阶段：最终结果用 save_dataset 导出，并明确 output_path。
6）清理阶段：任务结束后用 drop_frame 释放无用数据，避免内存增长。

安全与质量规则：
- 不得臆造不存在的字段，必须先验证列名是否存在。
- 工具调用失败时要说明原因，并给出下一步可执行的工具调用建议。
- 输出结论必须包含“做了什么、影响了多少行/列、当前状态如何”。
""".strip()


__all__ = [
    "FRAME_STORE",
    "AGENT_TOOLS",
    "DATA_AGENT_SYSTEM_PROMPT",
    "list_frames",
    "drop_frame",
    "load_dataset",
    "get_dataset_info",
    "get_preview",
    "clean_missing_values",
    "convert_currency_column",
    "add_tier_column",
    "filter_rows",
    "detect_outliers_iqr",
    "save_dataset",
]
