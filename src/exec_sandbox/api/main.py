"""FastAPI application assembly for exec-sandbox.

Creates and configures the unified FastAPI app that serves both the REST API
and the MCP SSE transport.  Uses an async context-manager lifespan to start
and stop the global SessionManager.

Usage::

    uvicorn exec_sandbox.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from exec_sandbox.api import routes_mcp, routes_rest
from exec_sandbox.api.manager import SessionManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global manager (one per process)
# ---------------------------------------------------------------------------

manager: SessionManager = SessionManager()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Start the SessionManager on startup and stop it on shutdown."""
    await manager.start()
    routes_rest.set_manager(manager)
    routes_mcp.set_manager(manager)
    logger.info("exec-sandbox API ready")
    yield
    await manager.stop()
    logger.info("exec-sandbox API stopped")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app: FastAPI = FastAPI(
    title="exec-sandbox API",
    description=("Secure code execution in isolated QEMU microVMs. Exposes a REST API and an MCP SSE transport."),
    version="0.0.0.dev0",
    lifespan=_lifespan,
)

app.include_router(routes_rest.router)
app.include_router(routes_mcp.router)
