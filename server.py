import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from panrag import build_agentic
from panrag.memory import EphemeralMemory
from panrag import config
from panrag.logging_utils import log_event

chain = None
mem = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain, mem
    try:
        chain = build_agentic()
    except Exception as exc:
        chain = None
        app.logger = getattr(app, "logger", None)
        if app.logger:
            app.logger.error(f"Failed to build chain at startup: {exc}")
    # Initialize ephemeral memory (best-effort)
    try:
        mem = EphemeralMemory()
    except Exception:
        mem = None
    try:
        log_event(
            "server.start",
            {
                "mem_ready": bool(mem),
                "answer_use_history": bool(getattr(config, "ANSWER_USE_HISTORY", True)),
                "answer_history_turns": int(getattr(config, "ANSWER_HISTORY_TURNS", 0)),
            },
        )
    except Exception:
        pass
    yield


app = FastAPI(title="RAG Server", version="1.0.0", lifespan=lifespan)

# Static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")


class QueryRequest(BaseModel):
    question: str
    conv_id: Optional[str] = "default"


class QueryResponse(BaseModel):
    answer: str
    conv_id: Optional[str] = None
    history_turns_used: int = 0


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
        # Ensure memory available and backfill if needed
        conv_id = (req.conv_id or "default").strip() or "default"
        conversation_history = []

        if mem is not None:
            try:
                mem.ensure_backfill(conv_id)
                # Get conversation history before adding new turn
                conversation_history = mem.get_buffer(conv_id)
                try:
                    log_event(
                        "mem.history.read",
                        {"conv_id": conv_id, "count": len(
                            conversation_history)},
                    )
                except Exception:
                    pass
                mem.append_turn(
                    conv_id=conv_id,
                    role="user",
                    text=req.question,
                    meta={"tokens": len(req.question.split())},
                )
            except Exception:
                pass

        answer = chain.invoke_with_history(
            req.question, conversation_history, chain._common_meta(req.question))

        # Derive how many turns were used (mirror AnswerAgent selection)
        turns_used = 0
        try:
            if conversation_history and config.ANSWER_USE_HISTORY:
                window = max(0, int(config.ANSWER_HISTORY_TURNS))
                sel = conversation_history[-window:
                                           ] if window > 0 else conversation_history
                turns_used = len(sel)
        except Exception:
            turns_used = 0
        try:
            log_event(
                "mem.history.used",
                {"conv_id": conv_id, "turns_used": int(turns_used)},
            )
        except Exception:
            pass

        if mem is not None:
            try:
                text = str(answer)
                mem.append_turn(
                    conv_id=conv_id,
                    role="assistant",
                    text=text,
                    meta={"tokens": len(text.split())},
                )
            except Exception:
                pass

        return QueryResponse(answer=str(answer), conv_id=conv_id, history_turns_used=turns_used)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
