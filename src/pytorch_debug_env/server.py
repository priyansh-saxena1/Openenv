# src/pytorch_debug_env/server.py
from fastapi import FastAPI, Query
from uuid import uuid4

from .environment import PyTorchDebugEnv
from .models import PyTorchDebugAction
from .scenario_generator import ScenarioGenerator
from .bug_library import BUG_TEMPLATES

app = FastAPI(title="PyTorch Debug Env")

sessions = {}
latest_session_id = None

@app.get("/")
async def root():
    return {
        "name": "pytorch-debug-env",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/health"],
        "tasks": ["easy", "medium", "hard"]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(task_id: str = "easy", seed: int | None = None):
    global latest_session_id
    session_id = str(uuid4())
    env = PyTorchDebugEnv(generator=ScenarioGenerator(BUG_TEMPLATES))
    sessions[session_id] = env
    latest_session_id = session_id
    obs = await env.reset(task_id=task_id, seed=seed)
    return {"session_id": session_id, "observation": obs, "done": False}


@app.post("/step")
async def step(action: PyTorchDebugAction, session_id: str = Query(None)):
    sid = session_id or latest_session_id
    env = sessions.get(sid)
    if not env:
        return {"error": "Invalid session_id"}
    return await env.step(action)


@app.get("/state")
async def state(session_id: str = Query(None)):
    sid = session_id or latest_session_id
    env = sessions.get(sid)
    if not env:
        return {"error": "Invalid session_id"}
    return await env.state()
