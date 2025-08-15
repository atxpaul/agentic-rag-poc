import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from panrag import build_agentic

chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain
    try:
        chain = build_agentic()
    except Exception as exc:
        chain = None
        app.logger = getattr(app, "logger", None)
        if app.logger:
            app.logger.error(f"Failed to build chain at startup: {exc}")
    yield


app = FastAPI(title="RAG Server", version="1.0.0", lifespan=lifespan)

# Static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    status = "ok" if chain is not None else "degraded"
    return {"status": status}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    global chain
    if chain is None:
        try:
            chain = build_agentic()
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Chain unavailable: {exc}")

    try:
        answer = chain.invoke(req.question)
        return QueryResponse(answer=str(answer))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
