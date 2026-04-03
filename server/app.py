import os

import uvicorn

try:
    from src.pytorch_debug_env.server import app
except ModuleNotFoundError:
    from pytorch_debug_env.server import app


def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)
