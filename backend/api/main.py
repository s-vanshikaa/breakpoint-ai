from fastapi import FastAPI
from pydantic import BaseModel

from models.ollama_client import ollama_client

app = FastAPI(title="BreakPoint AI")


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
