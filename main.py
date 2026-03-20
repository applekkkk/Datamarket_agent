from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from service import load_file, process as run_process

app = FastAPI(title="Data Agent Service")


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
        raise HTTPException(status_code=500, detail=f"????: {exc}") from exc


@app.post("/download")
async def download(user_id: int = Form(...)):
    csv_path = Path(__file__).resolve().parent / "data" / f"{user_id}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="????????")
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"{user_id}.csv",
    )
