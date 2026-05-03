"""FastAPI + MCP server for exec-sandbox.

Exposes a unified HTTP application with:
- REST API at /api/v1 for sandbox lifecycle management and code execution.
- MCP (Model Context Protocol) SSE transport at /mcp for AI-agent tool use.

Both layers orchestrate QEMU microVMs via the Scheduler and Session APIs.

Usage::

    uvicorn exec_sandbox.api.main:app --host 0.0.0.0 --port 8000

Or programmatically::

    from exec_sandbox.api.main import app
    import uvicorn
    uvicorn.run(app)
"""

from exec_sandbox.api.main import app

__all__ = ["app"]
