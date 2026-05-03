"""MCP (Model Context Protocol) SSE transport routes for exec-sandbox.

Implements the MCP Server-Sent Events transport on top of FastAPI, exposing
sandbox operations as MCP tools for AI agents.

Endpoints:
    GET  /mcp/sse       Establish an SSE connection; sends an ``endpoint`` event
                        with the URL the client must POST messages to.
    POST /mcp/messages  Receive client-to-server MCP messages.

Tools exposed:
    sandbox_create      Create a new sandbox session.
    sandbox_exec        Execute code in a session.
    sandbox_write_file  Write a file into a session.
    sandbox_read_file   Read a file from a session.
    sandbox_list_files  List files in a session.
    sandbox_destroy     Destroy a session.
"""

from __future__ import annotations

import io
import json
import logging
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types
from fastapi import APIRouter, Request, Response
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.server.sse import SseServerTransport

if TYPE_CHECKING:
    from exec_sandbox.api.manager import SessionManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router and MCP server singletons
# ---------------------------------------------------------------------------

router: APIRouter = APIRouter(prefix="/mcp", tags=["mcp"])

mcp_server: Server = Server("exec-sandbox")

_sse_transport: SseServerTransport = SseServerTransport("/mcp/messages")

# The manager is injected at startup from main.py via set_manager().
_manager: SessionManager | None = None


def set_manager(manager: SessionManager) -> None:
    """Attach the global SessionManager (called from app lifespan)."""
    global _manager  # noqa: PLW0603
    _manager = manager


def _get_manager() -> SessionManager:
    """Return the global SessionManager, raising if not initialised."""
    if _manager is None:  # pragma: no cover
        raise RuntimeError("SessionManager not initialised")
    return _manager


# ---------------------------------------------------------------------------
# MCP Tool definitions
# ---------------------------------------------------------------------------


@mcp_server.list_tools()
async def list_tools_handler() -> list[mcp_types.Tool]:
    """Return the list of tools available to MCP clients."""
    return [
        mcp_types.Tool(
            name="sandbox_create",
            description="Create a new isolated sandbox session. Returns a session_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "language": {"type": "string", "description": "Programming language (e.g. 'python')."},
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Packages to pre-install.",
                    },
                },
                "required": ["language"],
            },
        ),
        mcp_types.Tool(
            name="sandbox_exec",
            description="Execute code in an existing sandbox session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session UUID from sandbox_create."},
                    "code": {"type": "string", "description": "Source code to execute."},
                },
                "required": ["session_id", "code"],
            },
        ),
        mcp_types.Tool(
            name="sandbox_write_file",
            description="Write a file into a sandbox session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "path": {"type": "string", "description": "Relative path inside the sandbox."},
                    "content": {"type": "string", "description": "UTF-8 file content."},
                },
                "required": ["session_id", "path", "content"],
            },
        ),
        mcp_types.Tool(
            name="sandbox_read_file",
            description="Read a file from a sandbox session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "path": {"type": "string", "description": "Relative path inside the sandbox."},
                },
                "required": ["session_id", "path"],
            },
        ),
        mcp_types.Tool(
            name="sandbox_list_files",
            description="List files in a directory inside a sandbox session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "path": {"type": "string", "description": "Directory path (empty string for root)."},
                },
                "required": ["session_id"],
            },
        ),
        mcp_types.Tool(
            name="sandbox_destroy",
            description="Destroy a sandbox session and free its resources.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                },
                "required": ["session_id"],
            },
        ),
    ]


async def _tool_sandbox_create(arguments: dict[str, Any], manager: SessionManager) -> dict[str, Any]:
    """Handle sandbox_create tool call."""
    language: str = arguments.get("language", "python")
    packages: list[str] = arguments.get("packages", [])
    session_id = await manager.create_session(language=language, packages=packages)
    return {"session_id": session_id}


async def _tool_sandbox_exec(arguments: dict[str, Any], manager: SessionManager) -> dict[str, Any]:
    """Handle sandbox_exec tool call."""
    session_id = str(arguments["session_id"])
    code = str(arguments["code"])
    try:
        session = manager.get_session(session_id)
    except KeyError:
        return {"error": f"Session '{session_id}' not found"}
    exec_result = await session.exec(code)
    return {
        "stdout": exec_result.stdout,
        "stderr": exec_result.stderr,
        "exit_code": exec_result.exit_code,
    }


async def _tool_sandbox_write_file(arguments: dict[str, Any], manager: SessionManager) -> dict[str, Any]:
    """Handle sandbox_write_file tool call."""
    session_id = str(arguments["session_id"])
    path = str(arguments["path"])
    content = str(arguments["content"])
    try:
        session = manager.get_session(session_id)
    except KeyError:
        return {"error": f"Session '{session_id}' not found"}
    await session.write_file(path, content.encode("utf-8"))
    return {"success": True}


async def _tool_sandbox_read_file(arguments: dict[str, Any], manager: SessionManager) -> dict[str, Any]:
    """Handle sandbox_read_file tool call."""
    session_id = str(arguments["session_id"])
    path = str(arguments["path"])
    try:
        session = manager.get_session(session_id)
    except KeyError:
        return {"error": f"Session '{session_id}' not found"}
    buf = io.BytesIO()
    await session.read_file(path, destination=buf)
    return {"content": buf.getvalue().decode("utf-8")}


async def _tool_sandbox_list_files(arguments: dict[str, Any], manager: SessionManager) -> dict[str, Any]:
    """Handle sandbox_list_files tool call."""
    session_id = str(arguments["session_id"])
    path = str(arguments.get("path", ""))
    try:
        session = manager.get_session(session_id)
    except KeyError:
        return {"error": f"Session '{session_id}' not found"}
    entries = await session.list_files(path)
    return {"files": [{"name": e.name, "is_dir": e.is_dir, "size": e.size} for e in entries]}


async def _tool_sandbox_destroy(arguments: dict[str, Any], manager: SessionManager) -> dict[str, Any]:
    """Handle sandbox_destroy tool call."""
    session_id = str(arguments["session_id"])
    try:
        await manager.destroy_session(session_id)
    except KeyError:
        return {"error": f"Session '{session_id}' not found"}
    return {"success": True}


_TOOL_DISPATCH: dict[str, Any] = {
    "sandbox_create": _tool_sandbox_create,
    "sandbox_exec": _tool_sandbox_exec,
    "sandbox_write_file": _tool_sandbox_write_file,
    "sandbox_read_file": _tool_sandbox_read_file,
    "sandbox_list_files": _tool_sandbox_list_files,
    "sandbox_destroy": _tool_sandbox_destroy,
}


@mcp_server.call_tool()
async def call_tool_handler(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
    """Dispatch MCP tool calls to the appropriate sandbox operation."""
    manager = _get_manager()
    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        result: dict[str, Any] = {"error": f"Unknown tool: {name}"}
    else:
        result = await handler(arguments, manager)
    return [mcp_types.TextContent(type="text", text=json.dumps(result))]


# ---------------------------------------------------------------------------
# SSE transport endpoints
# ---------------------------------------------------------------------------


@router.get("/sse")
async def sse_endpoint(request: Request) -> Response:
    """Establish an SSE connection for MCP communication.

    The client connects here to receive the ``endpoint`` event containing
    the URL it must POST messages to.  Subsequent MCP messages arrive via
    Server-Sent Events on this same connection.
    """
    async with _sse_transport.connect_sse(request.scope, request.receive, request._send) as (  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        read_stream,
        write_stream,
    ):
        init_options = mcp_server.create_initialization_options(
            notification_options=NotificationOptions(),
        )
        await mcp_server.run(read_stream, write_stream, init_options)
    # connect_sse handles the full response internally; return a dummy response
    # that FastAPI's routing infrastructure never actually sends.
    return Response()  # pragma: no cover


@router.post("/messages")
async def messages_endpoint(request: Request) -> Response:
    """Receive a client-to-server MCP message.

    Called by the MCP client after parsing the ``endpoint`` URL from the SSE
    ``endpoint`` event.  Routes the message to the appropriate open session.
    """
    await _sse_transport.handle_post_message(request.scope, request.receive, request._send)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    return Response()  # pragma: no cover
