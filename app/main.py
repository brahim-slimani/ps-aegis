"""uvicorn app.api.main:app --port 8000

"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.api import routes_ask
from app.backend_mock import routes_mock
from app.backend_mock import service

def create_app(db_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = db_path or get_settings().backend_db
        routes_mock._initialise(path)
        app.state.db_path = path
        yield

    app = FastAPI(
        title="PS RAG Enterprise Backend", 
        version=get_settings().backend_version, 
        description=routes_mock.DESCRIPTION,
        openapi_tags=routes_mock.TAGS,
        lifespan=lifespan,
    )

    # ---- domain errors -> HTTP ----
    def _handler(code: int):
        async def handle(_: Request, exc: Exception):
            return JSONResponse(status_code=code, content={"detail": str(exc)})
        return handle

    app.add_exception_handler(service.NotFound, _handler(404))
    app.add_exception_handler(service.Conflict, _handler(409))
    app.add_exception_handler(service.Invalid, _handler(422))

    app.include_router(routes_ask.getRagRouter())
    app.include_router(routes_mock.getBackendMockRouter())
    return app

def _default_app() -> FastAPI:
    return create_app()

app = _default_app()