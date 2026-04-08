from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.results import router as results_router
from config import settings
from models.ollama_client import (
    OllamaError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
    ollama_client,
)

app = FastAPI(title="BreakPoint AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(results_router)


@app.exception_handler(OllamaError)
async def ollama_error_handler(request: Request, exc: OllamaError) -> JSONResponse:
    unavailable = isinstance(exc, (OllamaUnavailableError, OllamaModelNotFoundError))
    return JSONResponse(status_code=503 if unavailable else 502, content={"detail": str(exc)})


class CompletionRequest(BaseModel):
    prompt: str
    system: str | None = None


class CompletionResponse(BaseModel):
    response: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/complete", response_model=CompletionResponse)
async def complete(request: CompletionRequest):
    response = await ollama_client.complete(request.prompt, request.system)
    return CompletionResponse(response=response)
