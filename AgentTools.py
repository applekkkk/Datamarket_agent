"""
Data Agent Tools — 面向结构化表格数据的处理工具集
按生命周期阶段组织，配合 DATA_AGENT_SYSTEM_PROMPT 使用
"""
import json
from datetime import datetime
from functools import wraps
from typing import Any, Optional

import pandas as pd
from langchain_core.tools import tool


class FrameRegistry:
    """单例：管理所有 frame、操作日志、快照。"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._frames: dict[str, pd.DataFrame] = {}
            cls._instance._snapshots: dict[str, pd.DataFrame] = {}
            cls._instance._op_log: list[dict] = []
        return cls._instance

    def put(self, frame_id: str, df: pd.DataFrame):
        self._frames[frame_id] = df

    def get(self, frame_id: str) -> pd.DataFrame:
        if frame_id not in self._frames:
            raise KeyError(f"frame '{frame_id}' 不存在，请先调用 load_dataset 或检查 list_frames")
        return self._frames[frame_id]

    def drop(self, frame_id: str):
        self._frames.pop(frame_id, None)
        self._snapshots.pop(frame_id, None)

    def list(self) -> list[str]:
        return list(self._frames.keys())

    def save_snapshot(self, frame_id: str):
        self._snapshots[frame_id] = self._frames[frame_id].copy(deep=True)

    def get_snapshot(self, frame_id: str) -> pd.DataFrame | None:
        return self._snapshots.get(frame_id)

    def log(self, tool_name: str, params: dict, result_summary: str):
        self._op_log.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "tool": tool_name,
            "params": params,
            "summary": result_summary,
        })

    def get_log(self) -> list[dict]:
        return list(self._op_log)

    def clear_log(self):
        self._op_log.clear()


_reg = FrameRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# 1. 加载阶段
# ─────────────────────────────────────────────────────────────────────────────

@tool(description=(
    "从本地路径加载 CSV，返回 frame_id 供后续工具使用。"
    "参数：path，encoding（默认 utf-8），sep（默认逗号）。"
    "返回：{frame_id, rows, cols, columns}。"
    "加载后必须调用 get_dataset_info 了解数据质量再做处理。"
))
def load_dataset(path: str, encoding: str = "utf-8", sep: str = ",") -> dict:
    df = pd.read_csv(path, encoding=encoding, sep=sep)
    frame_id = f"frame_{datetime.now().strftime('%H%M%S%f')[:10]}"
    _reg.put(frame_id, df)
    summary = f"加载 {len(df)} 行 × {len(df.columns)} 列"
    _reg.log("load_dataset", {"path": path}, summary)
    return {"frame_id": frame_id, "rows": len(df), "cols": len(df.columns), "columns": df.columns.tolist()}


@tool(description=(
    "列出内存中所有 frame_id 及其规模。无参数。"
    "返回：[{frame_id, rows, cols}]。"
))
def list_frames() -> list[dict]:
    return [
        {"frame_id": fid, "rows": len(_reg.get(fid)), "cols": len(_reg.get(fid).columns)}
        for fid in _reg.list()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. 分析阶段
# ─────────────────────────────────────────────────────────────────────────────

@tool(description=(
    "获取 frame 的结构摘要：字段名、类型、空值数和空值率。"
    "参数：frame_id。"
    "返回：{rows, cols, columns:[{name,dtype,null_count,null_pct}]}。"
    "每次处理前必须先调用，不得臆造字段名。"
))
def get_dataset_info(frame_id: str) -> dict:
    df = _reg.get(frame_id)
    cols_info = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        cols_info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2) if len(df) else 0,
        })
    return {"rows": len(df), "cols": len(df.columns), "columns": cols_info}


@tool(description=(
    "返回 frame 的前 N 行样本，用于感知数据内容和格式。"
    "参数：frame_id，n（默认 10）。"
    "返回：{columns, rows:[...]}，null 值显示为 null。"
))
def get_preview(frame_id: str, n: int = 10) -> dict:
    df = _reg.get(frame_id)
    sample = df.head(n).where(pd.notna(df.head(n)), other=None)
    return {"columns": df.columns.tolist(), "rows": sample.values.tolist()}


@tool(description=(
    "统计指定列的枚举值频次分布，适合分析分类字段。"
    "参数：frame_id，column，top_n（默认 20）。"
    "返回：{column, unique_count, top_values:[{value,count,pct}]}。"
))
def get_value_counts(frame_id: str, column: str, top_n: int = 20) -> dict:
    df = _reg.get(frame_id)
    _assert_column(df, column)
    vc = df[column].value_counts(dropna=False).head(top_n)
    total = len(df)
    return {
        "column": column,
        "unique_count": int(df[column].nunique(dropna=False)),
        "top_values": [
            {"value": str(v), "count": int(c), "pct": round(c / total * 100, 2)}
            for v, c in vc.items()
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. 处理阶段
# ─────────────────────────────────────────────────────────────────────────────

@tool(description=(
    "按条件过滤行，保留满足条件的行。"
    "参数：frame_id，column，operator（eq/ne/gt/lt/gte/lte/isnull/notnull/contains），value（isnull/notnull 时忽略）。"
    "返回：{rows_before, rows_after, rows_removed}。"
))
def filter_rows(frame_id: str, column: str, operator: str, value: Any = None) -> dict:
    df = _reg.get(frame_id)
    _assert_column(df, column)
    before = len(df)
    ops = {
        "eq": df[column] == value, "ne": df[column] != value,
        "gt": df[column] > value,  "lt": df[column] < value,
        "gte": df[column] >= value, "lte": df[column] <= value,
        "isnull": df[column].isna(), "notnull": df[column].notna(),
        "contains": df[column].astype(str).str.contains(str(value), na=False),
    }
    if operator not in ops:
        raise ValueError(f"不支持的 operator '{operator}'，可选：{list(ops)}")
    result = df[ops[operator]].reset_index(drop=True)
    _reg.put(frame_id, result)
    summary = f"filter_rows: {before} → {len(result)} 行，删除 {before - len(result)} 行"
    _reg.log("filter_rows", {"frame_id": frame_id, "column": column, "operator": operator, "value": value}, summary)
    return {"rows_before": before, "rows_after": len(result), "rows_removed": before - len(result)}


@tool(description=(
    "删除重复行，保留第一次出现的记录。"
    "参数：frame_id，subset（判重列名列表，默认全列）。"
    "返回：{rows_before, rows_after, duplicates_removed}。"
))
def drop_duplicates(frame_id: str, subset: list[str] | None = None) -> dict:
    df = _reg.get(frame_id)
    if subset:
        for col in subset:
            _assert_column(df, col)
    before = len(df)
    result = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    _reg.put(frame_id, result)
    removed = before - len(result)
    _reg.log("drop_duplicates", {"frame_id": frame_id, "subset": subset}, f"删除 {removed} 条重复行")
    return {"rows_before": before, "rows_after": len(result), "duplicates_removed": removed}


@tool(description=(
    "对指定列的空值进行填充。"
    "参数：frame_id，column，strategy（fixed/mean/median/mode），value（strategy=fixed 时必填）。"
    "返回：{column, null_before, null_after, filled_count}。"
))
def fill_nulls(frame_id: str, column: str, strategy: str, value: Any = None) -> dict:
    df = _reg.get(frame_id)
    _assert_column(df, column)
    null_before = int(df[column].isna().sum())
    if strategy == "fixed":
        if value is None:
            raise ValueError("strategy=fixed 时 value 不能为空")
        df[column] = df[column].fillna(value)
    elif strategy == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif strategy == "median":
        df[column] = df[column].fillna(df[column].median())
    elif strategy == "mode":
        mode_val = df[column].mode()
        if not mode_val.empty:
            df[column] = df[column].fillna(mode_val[0])
    else:
        raise ValueError(f"不支持的 strategy '{strategy}'，可选：fixed/mean/median/mode")
    _reg.put(frame_id, df)
    null_after = int(df[column].isna().sum())
    _reg.log("fill_nulls", {"frame_id": frame_id, "column": column, "strategy": strategy},
             f"填充 {null_before - null_after} 个空值")
    return {"column": column, "null_before": null_before, "null_after": null_after,
            "filled_count": null_before - null_after}


@tool(description=(
    "将指定列转换为目标数据类型。"
    "参数：frame_id，column，target_type（int/float/str/bool/datetime）。"
    "转换失败的值变为 NaN/NaT，不中断执行。"
    "返回：{column, dtype_before, dtype_after, failed_count}。"
))
def cast_column(frame_id: str, column: str, target_type: str) -> dict:
    df = _reg.get(frame_id)
    _assert_column(df, column)
    dtype_before = str(df[column].dtype)
    type_map = {"int": "Int64", "float": "float64", "str": "string", "bool": "boolean"}
    if target_type not in {**type_map, "datetime": None}:
        raise ValueError(f"不支持的 target_type '{target_type}'，可选：int/float/str/bool/datetime")
    null_before = int(df[column].isna().sum())
    if target_type == "datetime":
        df[column] = pd.to_datetime(df[column], errors="coerce")
    else:
        df[column] = df[column].astype(type_map[target_type], errors="ignore")
    _reg.put(frame_id, df)
    null_after = int(df[column].isna().sum())
    _reg.log("cast_column", {"frame_id": frame_id, "column": column, "target_type": target_type},
             f"{column}: {dtype_before} → {target_type}，转换失败 {max(0, null_after - null_before)} 个")
    return {"column": column, "dtype_before": dtype_before, "dtype_after": str(df[column].dtype),
            "failed_count": max(0, null_after - null_before)}


@tool(description=(
    "批量重命名列。参数：frame_id，rename_map（{旧列名: 新列名}）。"
    "返回：{renamed}。"
))
def rename_columns(frame_id: str, rename_map: dict[str, str]) -> dict:
    df = _reg.get(frame_id)
    for old in rename_map:
        _assert_column(df, old)
    df = df.rename(columns=rename_map)
    _reg.put(frame_id, df)
    _reg.log("rename_columns", {"frame_id": frame_id, "rename_map": rename_map}, f"重命名 {rename_map}")
    return {"renamed": rename_map}


@tool(description=(
    "用 pandas eval 表达式派生新列或覆盖已有列。"
    "参数：frame_id，new_column（新列名），expression（如 'price * 7.24' 或 'col_a + col_b'）。"
    "只允许算术和比较运算，禁止函数调用，防止注入。"
    "返回：{new_column, sample_values（前5行）}。"
))
def apply_expression(frame_id: str, new_column: str, expression: str) -> dict:
    df = _reg.get(frame_id)
    for token in ["__", "import", "exec", "eval", "open", "("]:
        if token in expression:
            raise ValueError(f"表达式含不允许的内容 '{token}'，只支持算术和比较运算符")
    df[new_column] = df.eval(expression)
    _reg.put(frame_id, df)
    _reg.log("apply_expression", {"frame_id": frame_id, "new_column": new_column, "expression": expression},
             f"生成列 '{new_column}' = {expression}")
    return {"new_column": new_column, "sample_values": df[new_column].head(5).tolist()}


@tool(description=(
    "保留或删除指定列。参数：frame_id，columns（列名列表），mode（keep/drop，默认 keep）。"
    "返回：{cols_before, cols_after, columns_now}。"
))
def select_columns(frame_id: str, columns: list[str], mode: str = "keep") -> dict:
    df = _reg.get(frame_id)
    for col in columns:
        _assert_column(df, col)
    cols_before = len(df.columns)
    df = df[columns] if mode == "keep" else df.drop(columns=columns)
    _reg.put(frame_id, df)
    _reg.log("select_columns", {"frame_id": frame_id, "columns": columns, "mode": mode},
             f"({mode}): {cols_before} → {len(df.columns)} 列")
    return {"cols_before": cols_before, "cols_after": len(df.columns), "columns_now": df.columns.tolist()}


@tool(description=(
    "按指定列排序行。参数：frame_id，by（列名或列表），ascending（默认 True）。"
    "返回：{rows, sorted_by}。"
))
def sort_rows(frame_id: str, by: str | list[str], ascending: bool = True) -> dict:
    df = _reg.get(frame_id)
    by_list = [by] if isinstance(by, str) else by
    for col in by_list:
        _assert_column(df, col)
    df = df.sort_values(by=by_list, ascending=ascending).reset_index(drop=True)
    _reg.put(frame_id, df)
    _reg.log("sort_rows", {"frame_id": frame_id, "by": by, "ascending": ascending},
             f"按 {by} {'升序' if ascending else '降序'}")
    return {"rows": len(df), "sorted_by": by_list}


@tool(description=(
    "将两个 frame 按指定列做 join，产生新 frame。"
    "参数：left_frame_id，right_frame_id，on（key 列名或列表），how（inner/left/right/outer，默认 inner），output_frame_id。"
    "返回：{output_frame_id, rows, cols}。"
))
def merge_frames(left_frame_id: str, right_frame_id: str, on: str | list[str],
                 how: str = "inner", output_frame_id: str = "merged") -> dict:
    left = _reg.get(left_frame_id)
    right = _reg.get(right_frame_id)
    result = pd.merge(left, right, on=on, how=how)
    _reg.put(output_frame_id, result)
    _reg.log("merge_frames", {"left": left_frame_id, "right": right_frame_id, "on": on, "how": how},
             f"{left_frame_id} × {right_frame_id} → {output_frame_id}，{len(result)} 行")
    return {"output_frame_id": output_frame_id, "rows": len(result), "cols": len(result.columns)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. 校验阶段
# ─────────────────────────────────────────────────────────────────────────────

@tool(description=(
    "校验 frame 是否符合预期 schema：列名存在性、类型是否匹配。"
    "参数：frame_id，expected_schema（[{name, dtype}]，dtype 可为 int/float/str/bool/datetime）。"
    "返回：{passed, issues:[{column, issue}]}。"
))
def validate_schema(frame_id: str, expected_schema: list[dict]) -> dict:
    df = _reg.get(frame_id)
    dtype_groups = {
        "int": ["int64", "Int64", "int32"], "float": ["float64", "float32"],
        "str": ["object", "string"], "bool": ["bool", "boolean"],
        "datetime": ["datetime64[ns]"],
    }
    issues = []
    for spec in expected_schema:
        col, expected_type = spec.get("name"), spec.get("dtype")
        if col not in df.columns:
            issues.append({"column": col, "issue": "列不存在"})
            continue
        actual = str(df[col].dtype)
        if actual not in dtype_groups.get(expected_type, [expected_type]):
            issues.append({"column": col, "issue": f"类型不匹配，期望 {expected_type}，实际 {actual}"})
    return {"passed": len(issues) == 0, "issues": issues}


@tool(description=(
    "生成各列空值统计报告，只返回有空值的列。"
    "参数：frame_id。返回：{total_rows, null_summary:[{column,null_count,null_pct}]}。"
))
def get_null_report(frame_id: str) -> dict:
    df = _reg.get(frame_id)
    report = []
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n > 0:
            report.append({"column": col, "null_count": n, "null_pct": round(n / len(df) * 100, 2)})
    return {"total_rows": len(df), "null_summary": sorted(report, key=lambda x: -x["null_count"])}


@tool(description=(
    "对比两个 frame 的行列变化，用于处理前后的差异确认。"
    "参数：before_frame_id，after_frame_id。"
    "返回：{rows_before, rows_after, rows_delta, cols_before, cols_after, new_cols, removed_cols, null_delta}。"
    "处理后调用此工具可自动满足'影响了多少行/列'的输出要求。"
))
def diff_frames(before_frame_id: str, after_frame_id: str) -> dict:
    before = _reg.get(before_frame_id)
    after = _reg.get(after_frame_id)
    before_cols, after_cols = set(before.columns), set(after.columns)
    return {
        "rows_before": len(before), "rows_after": len(after),
        "rows_delta": len(after) - len(before),
        "cols_before": len(before.columns), "cols_after": len(after.columns),
        "new_cols": sorted(after_cols - before_cols),
        "removed_cols": sorted(before_cols - after_cols),
        "null_delta": int(after.isna().sum().sum()) - int(before.isna().sum().sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. 保存阶段
# ─────────────────────────────────────────────────────────────────────────────

@tool(description=(
    "将 frame 导出到本地文件。"
    "参数：frame_id，output_path，format（csv/json/parquet，默认 csv），index（默认 False）。"
    "返回：{output_path, format, rows, cols}。"
))
def save_dataset(frame_id: str, output_path: str, format: str = "csv", index: bool = False) -> dict:
    df = _reg.get(frame_id)
    if format == "csv":
        df.to_csv(output_path, index=index)
    elif format == "json":
        df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    elif format == "parquet":
        df.to_parquet(output_path, index=index)
    else:
        raise ValueError(f"不支持的格式 '{format}'，可选：csv/json/parquet")
    _reg.log("save_dataset", {"frame_id": frame_id, "output_path": output_path, "format": format},
             f"{frame_id} → {output_path}（{format}）")
    return {"output_path": output_path, "format": format, "rows": len(df), "cols": len(df.columns)}


@tool(description=(
    "将操作日志和数据摘要导出为 JSON 或 Markdown 报告。"
    "参数：frame_id，output_path，format（json/markdown，默认 markdown）。"
    "报告包含：数据基本信息、字段列表、操作历史、空值摘要。"
    "返回：{output_path, steps_count}。"
))
def export_report(frame_id: str, output_path: str, format: str = "markdown") -> dict:
    df = _reg.get(frame_id)
    log = _reg.get_log()
    info = {
        "frame_id": frame_id, "rows": len(df), "cols": len(df.columns),
        "columns": [{"name": c, "dtype": str(df[c].dtype), "null_count": int(df[c].isna().sum())}
                    for c in df.columns],
        "operations": log,
    }
    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    else:
        lines = [
            f"# 数据处理报告 — {frame_id}",
            f"\n生成时间：{datetime.now().isoformat(timespec='seconds')}",
            f"\n## 数据概况\n- 行数：{len(df)}\n- 列数：{len(df.columns)}",
            "\n## 字段信息\n| 字段 | 类型 | 空值数 |\n| --- | --- | --- |",
        ]
        for c in info["columns"]:
            lines.append(f"| {c['name']} | {c['dtype']} | {c['null_count']} |")
        lines.append("\n## 操作历史")
        for i, op in enumerate(log, 1):
            lines.append(f"{i}. **{op['tool']}** ({op['time']})：{op['summary']}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return {"output_path": output_path, "steps_count": len(log)}


# ─────────────────────────────────────────────────────────────────────────────
# 6. 清理阶段
# ─────────────────────────────────────────────────────────────────────────────

@tool(description=(
    "释放指定 frame 占用的内存，同时删除其快照。"
    "参数：frame_id。任务结束后必须调用，避免内存增长。"
    "返回：{dropped, remaining_frames}。"
))
def drop_frame(frame_id: str) -> dict:
    _reg.drop(frame_id)
    _reg.log("drop_frame", {"frame_id": frame_id}, f"释放 {frame_id}")
    return {"dropped": frame_id, "remaining_frames": _reg.list()}


@tool(description=(
    "释放全部 frame 和快照，并清空操作日志。整个会话结束后调用。"
    "无参数。返回：{cleared_count}。"
))
def clear_all_frames() -> dict:
    count = len(_reg.list())
    for fid in list(_reg.list()):
        _reg.drop(fid)
    _reg.clear_log()
    return {"cleared_count": count}


# ─────────────────────────────────────────────────────────────────────────────
# 横切工具（任意阶段均可调用）
# ─────────────────────────────────────────────────────────────────────────────

@tool(description=(
    "深拷贝一个 frame，用于在重要处理前保留原始状态。"
    "参数：frame_id，new_frame_id（默认在原名后加 '_clone'）。"
    "最佳实践：处理前 clone，失败时用 rollback_frame 恢复。"
    "返回：{original_frame_id, new_frame_id, rows, cols}。"
))
def clone_frame(frame_id: str, new_frame_id: str | None = None) -> dict:
    df = _reg.get(frame_id)
    new_id = new_frame_id or f"{frame_id}_clone"
    _reg.put(new_id, df.copy(deep=True))
    _reg.save_snapshot(new_id)
    _reg.log("clone_frame", {"frame_id": frame_id, "new_frame_id": new_id}, f"{frame_id} → {new_id}")
    return {"original_frame_id": frame_id, "new_frame_id": new_id, "rows": len(df), "cols": len(df.columns)}


@tool(description=(
    "将 frame 回滚到最近一次 clone 保存的快照状态。"
    "参数：frame_id。若无快照则报错。"
    "返回：{frame_id, restored_rows, restored_cols}。"
))
def rollback_frame(frame_id: str) -> dict:
    snap = _reg.get_snapshot(frame_id)
    if snap is None:
        raise ValueError(f"frame '{frame_id}' 没有可用快照，请先调用 clone_frame")
    _reg.put(frame_id, snap.copy(deep=True))
    _reg.log("rollback_frame", {"frame_id": frame_id}, f"{frame_id} 已还原至快照")
    return {"frame_id": frame_id, "restored_rows": len(snap), "restored_cols": len(snap.columns)}


@tool(description=(
    "获取完整操作日志，每条含时间、工具名、参数、摘要。"
    "参数：last_n（只返回最近 N 条，默认全部）。"
    "返回：[{time, tool, params, summary}]。"
))
def get_operation_log(last_n: int | None = None) -> list[dict]:
    log = _reg.get_log()
    return log[-last_n:] if last_n else log


@tool(description=(
    "检查列名是否存在于 frame 中，并返回相似列名。"
    "参数：frame_id，column。"
    "返回：{exists, column, similar_columns}。"
    "Agent 操作任何列前应先调用此工具，避免臆造不存在的字段。"
))
def check_column_exists(frame_id: str, column: str) -> dict:
    df = _reg.get(frame_id)
    exists = column in df.columns
    similar = [c for c in df.columns if column.lower() in c.lower() or c.lower() in column.lower()]
    return {"exists": exists, "column": column, "similar_columns": similar}


@tool(description=(
    "对指定列做详细统计描述。数值列返回 count/mean/std/min/max/分位数，分类列返回 unique/top/freq。"
    "参数：frame_id，column。返回：{column, dtype, stats}。"
))
def describe_column(frame_id: str, column: str) -> dict:
    df = _reg.get(frame_id)
    _assert_column(df, column)
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        stats = {k: round(float(v), 4) for k, v in series.describe().items()}
    else:
        stats = {
            "count": int(series.count()), "unique": int(series.nunique()),
            "top": str(series.mode()[0]) if not series.mode().empty else None,
            "freq": int(series.value_counts().iloc[0]) if len(series.value_counts()) else 0,
        }
    return {"column": column, "dtype": str(series.dtype), "stats": stats}


# ─────────────────────────────────────────────────────────────────────────────
# 内部辅助
# ─────────────────────────────────────────────────────────────────────────────

def _assert_column(df: pd.DataFrame, column: str):
    if column not in df.columns:
        similar = [c for c in df.columns if column.lower() in c.lower()]
        hint = f"，相似列名：{similar}" if similar else ""
        raise KeyError(f"列 '{column}' 不存在{hint}，请先调用 check_column_exists 验证")


# ─────────────────────────────────────────────────────────────────────────────
# 工具注册表
# ─────────────────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    load_dataset, list_frames,                                          # 加载
    get_dataset_info, get_preview, get_value_counts,                    # 分析
    filter_rows, drop_duplicates, fill_nulls, cast_column,              # 处理
    rename_columns, apply_expression, select_columns, sort_rows,        # 处理
    merge_frames,                                                        # 处理
    validate_schema, get_null_report, diff_frames,                      # 校验
    save_dataset, export_report,                                        # 保存
    drop_frame, clear_all_frames,                                       # 清理
    clone_frame, rollback_frame, get_operation_log,                     # 横切
    check_column_exists, describe_column,                               # 横切
]
DATA_AGENT_SYSTEM_PROMPT = """
你是一个面向结构化表格数据的智能处理助手，目标是产出可复现、可追踪、可落地的数据结果。

数据生命周期规则：
1）加载阶段：先调用 load_dataset 读取数据并记录 frame_id。
2）分析阶段：处理前必须先调用 get_dataset_info / get_preview，明确字段、类型和空值情况。
3）处理阶段：每次只执行一个明确步骤，禁止一次做多个不透明变换。
4）校验阶段：每次处理后都要复核行数变化、空值变化和关键字段分布。
5）保存阶段：处理完成后用 save_dataset 输出结果（可覆盖原文件或保存新文件）。
6）总结阶段：必须输出一份 Markdown 报告，格式固定如下：

## 数据处理总结
### 1. 基本信息
- frame_id：
- 输入文件：
- 输出文件：
- 处理时间：

### 2. 执行步骤
1. 
2. 
3. 

### 3. 结果变化
- 行数变化：before -> after（减少/增加多少）
- 列数变化：before -> after
- 新增列：
- 删除列：
- 空值变化：before -> after

### 4. 数据质量评估
- 完整性：
- 一致性：
- 可用性：
- 风险项：

### 5. 后续建议
- 
- 

7）清理阶段：任务完成后调用 drop_frame / clear_all_frames 释放内存。

安全与质量规则：
- 不得臆造字段，引用列名前必须验证存在性。
- 工具失败时需解释原因，并给出下一步可执行工具建议。
- 所有结论必须包含“做了什么、影响多少行/列、当前状态如何”。
""".strip()



