from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline import run_pipeline

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"

app = FastAPI(title="TrustLayer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

async def save_upload(upload: Optional[UploadFile], case_dir: Path) -> Optional[str]:
    if upload is None or not upload.filename:
        return None
    destination = case_dir / Path(upload.filename).name
    destination.write_bytes(await upload.read())
    return str(destination)


@app.post("/api/analyze")
async def analyze(
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    claim: Optional[str] = Form(None),
):
    case_dir = UPLOAD_DIR / uuid4().hex
    case_dir.mkdir(parents=True, exist_ok=True)

    case = {
        "image": await save_upload(image, case_dir),
        "video": await save_upload(video, case_dir),
        "audio": await save_upload(audio, case_dir),
        "claim": claim,
    }
    return run_pipeline(case)