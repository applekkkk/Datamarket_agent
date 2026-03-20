from fastapi import BackgroundTasks, FastAPI, HTTPException

from service import load_file, Response

app = FastAPI(title="Data Agent Service")


@app.post("/agent/process")
async def process(file, user_id, user_prompt):
    file_path = load_file(file,user_id)
    report = await process(file_path,user_prompt)
    return Response(data=report)

@app.post("/download")
async def download(user_id):
    return
