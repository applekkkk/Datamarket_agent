import asyncio
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from service import load_file, process as run_process
from visualize import router

app = FastAPI(title="Data Agent Service")
app.include_router(router)

# Allow CORS for web/dev tools.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TASK_STORE = {}
TASK_TTL_SECONDS = 60 * 60


def _cleanup_tasks():
    now = time.time()
    expired = []
    for task_id, item in TASK_STORE.items():
        if item.get("status") in {"done", "failed"} and now - item.get("updated_at", now) > TASK_TTL_SECONDS:
            expired.append(task_id)
    for task_id in expired:
        TASK_STORE.pop(task_id, None)


async def _run_async_task(task_id: str, file_path: str, user_prompt: str):
    task = TASK_STORE.get(task_id)
    if not task:
        return

    task["status"] = "running"
    task["updated_at"] = time.time()

    try:
        result = await run_in_threadpool(run_process, file_path, user_prompt)
        task["status"] = "done"
        task["result"] = result
        task["updated_at"] = time.time()
    except Exception as exc:
        task["status"] = "failed"
        task["error"] = f"处理失败: {exc}"
        task["updated_at"] = time.time()


@app.post("/agent/start")
async def start_task(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    user_prompt: str = Form(...),
):
    try:
        _cleanup_tasks()
        file_path = load_file(file, user_id)
        task_id = uuid4().hex
        TASK_STORE[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "result": None,
            "error": "",
            "updated_at": time.time(),
        }
        asyncio.create_task(_run_async_task(task_id, file_path, user_prompt))
        return {"code": 200, "message": "success", "data": {"taskId": task_id, "status": "queued"}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"启动任务失败: {exc}") from exc


@app.get("/agent/status/{task_id}")
async def get_task_status(task_id: str):
    _cleanup_tasks()
    task = TASK_STORE.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    data = {
        "taskId": task_id,
        "status": task.get("status", "unknown"),
    }
    if task.get("status") == "done":
        data["result"] = task.get("result")
    elif task.get("status") == "failed":
        data["error"] = task.get("error") or "处理失败"

    return {"code": 200, "message": "success", "data": data}


@app.post("/agent/process")
async def process(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    user_prompt: str = Form(...),
):
    try:
        file_path = load_file(file, user_id)
        report = await run_in_threadpool(run_process, file_path, user_prompt)
        return {"code": 200, "message": "success", "data": report}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"处理失败: {exc}") from exc


@app.post("/download")
async def download(user_id: int = Form(...)):
    csv_path = Path(__file__).resolve().parent / "data" / f"{user_id}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="未找到处理后的文件")
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"{user_id}.csv",
    )
