import asyncio
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import load_settings
from .pipeline import ForecastPipeline

app = FastAPI(title="Forecast Agent API", version="0.1.0")

# 允许本地调试前端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 静态托管前端与输出
if Path("web").exists():
    app.mount("/web", StaticFiles(directory="web"), name="web")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


def _run_pipeline_for_file(file_path: Path) -> Dict[str, Any]:
    settings = load_settings()
    settings.data_path = file_path
    settings.output_dir = OUTPUT_DIR
    pipeline = ForecastPipeline(settings)
    output = pipeline.run()
    return {
        "metrics_table": output.metrics_table,
        "llm_analysis": output.llm_analysis,
        "chart_url": "/outputs/forecast_chart.html",
        "excel_url": "/outputs/predictions.xlsx",
    }


@app.post("/upload")
async def upload_excel(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="仅支持 Excel 文件 (.xlsx/.xls)")
    dest = DATA_DIR / "uploaded.xlsx"
    content = await file.read()
    dest.write_bytes(content)

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_pipeline_for_file, dest
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.get("/chart")
async def get_chart() -> FileResponse:
    chart_path = OUTPUT_DIR / "forecast_chart.html"
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="chart not found")
    return FileResponse(chart_path)


@app.get("/")
async def root() -> FileResponse:
    index_path = Path("web/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend not found")
    return FileResponse(index_path)

