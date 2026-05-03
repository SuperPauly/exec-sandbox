"""REST API routes for the exec-sandbox HTTP API.

Exposes sandbox lifecycle and code-execution operations at /api/v1.

Endpoints:
    POST   /api/v1/sandbox                          Create a new sandbox session.
    POST   /api/v1/sandbox/{session_id}/exec        Execute code in a session.
    POST   /api/v1/sandbox/{session_id}/file/write  Write a file into the sandbox.
    GET    /api/v1/sandbox/{session_id}/file/read   Read a file from the sandbox.
    GET    /api/v1/sandbox/{session_id}/files       List files in the sandbox.
    DELETE /api/v1/sandbox/{session_id}             Destroy a session.
"""

from __future__ import annotations

import io
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from exec_sandbox.api.manager import SessionManager

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter(prefix="/api/v1", tags=["sandbox"])

# ---------------------------------------------------------------------------
# Dependency injection helpers
# ---------------------------------------------------------------------------

_manager: SessionManager | None = None


def set_manager(manager: SessionManager) -> None:
    """Attach the global SessionManager (called from app lifespan)."""
    global _manager  # noqa: PLW0603
    _manager = manager


def get_manager() -> SessionManager:
    """FastAPI dependency that returns the global SessionManager."""
    if _manager is None:
        raise RuntimeError("SessionManager not initialised")  # pragma: no cover
    return _manager


# Typed dependency alias used in route signatures (avoids B008 warning).
ManagerDep = Annotated[SessionManager, Depends(get_manager)]

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateSandboxRequest(BaseModel):
    """Request body for POST /api/v1/sandbox."""

    language: str = Field(default="python", description="Programming language for the session.")
    packages: list[str] = Field(default_factory=list, description="Packages to pre-install.")


class CreateSandboxResponse(BaseModel):
    """Response body for POST /api/v1/sandbox."""

    session_id: str = Field(description="UUID identifying the new session.")


class ExecRequest(BaseModel):
    """Request body for POST /api/v1/sandbox/{session_id}/exec."""

    code: str = Field(description="Source code to execute in the sandbox.")


class ExecResponse(BaseModel):
    """Response body for POST /api/v1/sandbox/{session_id}/exec."""

    stdout: str = Field(description="Captured standard output.")
    stderr: str = Field(description="Captured standard error.")
    exit_code: int = Field(description="Process exit code (0 = success).")


class WriteFileRequest(BaseModel):
    """Request body for POST /api/v1/sandbox/{session_id}/file/write."""

    path: str = Field(description="Relative path inside the sandbox.")
    content: str = Field(description="UTF-8 encoded file content.")


class WriteFileResponse(BaseModel):
    """Response body for POST /api/v1/sandbox/{session_id}/file/write."""

    success: bool = Field(description="True on success.")


class ReadFileResponse(BaseModel):
    """Response body for GET /api/v1/sandbox/{session_id}/file/read."""

    content: str = Field(description="UTF-8 decoded file content.")


class FileInfoResponse(BaseModel):
    """A single file-system entry returned by the list endpoint."""

    name: str = Field(description="File or directory name.")
    is_dir: bool = Field(description="True if the entry is a directory.")
    size: int = Field(description="File size in bytes (0 for directories).")


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post("/sandbox", response_model=CreateSandboxResponse, status_code=201)
async def create_sandbox(
    body: CreateSandboxRequest,
    manager: ManagerDep,
) -> CreateSandboxResponse:
    """Create a new isolated sandbox session.

    Returns a ``session_id`` that must be passed to subsequent requests.
    """
    session_id = await manager.create_session(language=body.language, packages=body.packages)
    return CreateSandboxResponse(session_id=session_id)


@router.post("/sandbox/{session_id}/exec", response_model=ExecResponse)
async def exec_code(
    session_id: str,
    body: ExecRequest,
    manager: ManagerDep,
) -> ExecResponse:
    """Execute code inside an existing sandbox session.

    State persists across calls within the same session (variables, imports, …).
    """
    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found") from None

    result = await session.exec(body.code)
    return ExecResponse(stdout=result.stdout, stderr=result.stderr, exit_code=result.exit_code)


@router.post("/sandbox/{session_id}/file/write", response_model=WriteFileResponse)
async def write_file(
    session_id: str,
    body: WriteFileRequest,
    manager: ManagerDep,
) -> WriteFileResponse:
    """Write a file into the sandbox at the specified path."""
    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found") from None

    await session.write_file(body.path, body.content.encode("utf-8"))
    return WriteFileResponse(success=True)


@router.get("/sandbox/{session_id}/file/read", response_model=ReadFileResponse)
async def read_file(
    session_id: str,
    path: str,
    manager: ManagerDep,
) -> ReadFileResponse:
    """Read a file from the sandbox and return its UTF-8 decoded content."""
    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found") from None

    buf = io.BytesIO()
    await session.read_file(path, destination=buf)
    content = buf.getvalue().decode("utf-8")
    return ReadFileResponse(content=content)


@router.get("/sandbox/{session_id}/files", response_model=list[FileInfoResponse])
async def list_files(
    session_id: str,
    manager: ManagerDep,
    path: str = "",
) -> list[FileInfoResponse]:
    """List files and directories at ``path`` inside the sandbox."""
    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found") from None

    entries = await session.list_files(path)
    return [FileInfoResponse(name=e.name, is_dir=e.is_dir, size=e.size) for e in entries]


@router.delete("/sandbox/{session_id}", status_code=204)
async def destroy_sandbox(
    session_id: str,
    manager: ManagerDep,
) -> None:
    """Destroy a sandbox session and free its resources."""
    try:
        await manager.destroy_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found") from None
