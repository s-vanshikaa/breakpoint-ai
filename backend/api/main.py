from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.results import router as results_router
from models.ollama_client import ollama_client

app = FastAPI(title="BreakPoint AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(results_router)


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
