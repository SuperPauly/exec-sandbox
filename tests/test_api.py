"""Tests for the exec-sandbox FastAPI + MCP API layer.

All tests mock the underlying Scheduler and Session to avoid requiring
actual QEMU assets or hardware acceleration.  They exercise the HTTP
routing, request/response serialisation, dependency injection, error
handling, and SessionManager lifecycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from exec_sandbox.api import routes_mcp, routes_rest
from exec_sandbox.api.main import app
from exec_sandbox.api.manager import SessionManager
from exec_sandbox.models import ExecutionResult, FileInfo, TimingBreakdown

if TYPE_CHECKING:
    import io


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_exec_result(stdout: str = "", stderr: str = "", exit_code: int = 0) -> ExecutionResult:
    """Create a minimal ExecutionResult for mocking."""
    timing = TimingBreakdown(setup_ms=0, boot_ms=0, execute_ms=1, total_ms=1)
    return ExecutionResult(stdout=stdout, stderr=stderr, exit_code=exit_code, timing=timing)


def _make_session(
    *,
    exec_result: ExecutionResult | None = None,
    file_content: bytes = b"",
    file_list: list[FileInfo] | None = None,
) -> AsyncMock:
    """Build a mock Session with configurable return values."""
    session = AsyncMock()
    session.closed = False
    session.exec = AsyncMock(return_value=exec_result or _make_exec_result())
    session.write_file = AsyncMock(return_value=None)
    session.list_files = AsyncMock(return_value=file_list or [])
    session.close = AsyncMock(return_value=None)

    # read_file writes into the destination buffer
    async def _read_file(path: str, *, destination: io.BytesIO) -> None:
        destination.write(file_content)

    session.read_file = AsyncMock(side_effect=_read_file)
    return session


def _make_manager(session: AsyncMock | None = None) -> MagicMock:
    """Build a mock SessionManager."""
    mgr = MagicMock(spec=SessionManager)
    _session = session or _make_session()
    mgr.create_session = AsyncMock(return_value="test-session-id")
    mgr.get_session = MagicMock(return_value=_session)
    mgr.destroy_session = AsyncMock(return_value=None)
    mgr.start = AsyncMock(return_value=None)
    mgr.stop = AsyncMock(return_value=None)
    return mgr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session() -> AsyncMock:
    """A mock Session instance."""
    return _make_session()


@pytest.fixture
def mock_manager(mock_session: AsyncMock) -> MagicMock:
    """A mock SessionManager with the mock session attached."""
    return _make_manager(session=mock_session)


@pytest.fixture
async def client(mock_manager: MagicMock) -> AsyncClient:
    """AsyncClient pointed at the FastAPI app with mocked dependencies.

    Injects the mock_manager directly into the route modules so that no real
    Scheduler is started during the test.
    """
    routes_rest.set_manager(mock_manager)
    routes_mcp.set_manager(mock_manager)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# SessionManager unit tests
# ---------------------------------------------------------------------------


class TestSessionManager:
    """Unit tests for SessionManager (no HTTP layer)."""

    async def test_start_and_stop(self) -> None:
        """start() initialises the Scheduler; stop() shuts it down."""
        manager = SessionManager()
        mock_sched = MagicMock()
        mock_sched.__aenter__ = AsyncMock(return_value=mock_sched)
        mock_sched.__aexit__ = AsyncMock(return_value=None)

        with patch("exec_sandbox.api.manager.Scheduler", return_value=mock_sched):
            await manager.start()
            assert manager._scheduler is mock_sched
            await manager.stop()
            assert manager._scheduler is None

    async def test_stop_is_idempotent(self) -> None:
        """stop() can be called safely before start()."""
        manager = SessionManager()
        await manager.stop()  # Should not raise

    async def test_create_session_raises_if_not_started(self) -> None:
        """create_session() raises RuntimeError when Scheduler is not running."""
        manager = SessionManager()
        with pytest.raises(RuntimeError, match="not started"):
            await manager.create_session("python", [])

    async def test_create_and_get_session(self) -> None:
        """Sessions can be created and retrieved by ID."""
        manager = SessionManager()
        mock_session = _make_session()
        mock_sched = MagicMock()
        mock_sched.__aenter__ = AsyncMock(return_value=mock_sched)
        mock_sched.__aexit__ = AsyncMock(return_value=None)
        mock_sched.session = AsyncMock(return_value=mock_session)

        with patch("exec_sandbox.api.manager.Scheduler", return_value=mock_sched):
            await manager.start()
            session_id = await manager.create_session("python", [])
            assert len(session_id) == 32  # UUID4 hex has 32 chars
            retrieved = manager.get_session(session_id)
            assert retrieved is mock_session
            await manager.stop()

    async def test_get_session_raises_keyerror_for_unknown_id(self) -> None:
        """get_session() raises KeyError for an unknown session ID."""
        manager = SessionManager()
        with pytest.raises(KeyError):
            manager.get_session("nonexistent-id")

    async def test_destroy_session(self) -> None:
        """destroy_session() removes the session and closes it."""
        manager = SessionManager()
        mock_session = _make_session()
        mock_sched = MagicMock()
        mock_sched.__aenter__ = AsyncMock(return_value=mock_sched)
        mock_sched.__aexit__ = AsyncMock(return_value=None)
        mock_sched.session = AsyncMock(return_value=mock_session)

        with patch("exec_sandbox.api.manager.Scheduler", return_value=mock_sched):
            await manager.start()
            session_id = await manager.create_session("python", [])
            await manager.destroy_session(session_id)
            mock_session.close.assert_awaited_once()
            with pytest.raises(KeyError):
                manager.get_session(session_id)
            await manager.stop()

    async def test_destroy_session_raises_keyerror_for_unknown_id(self) -> None:
        """destroy_session() raises KeyError for an unknown session ID."""
        manager = SessionManager()
        with pytest.raises(KeyError):
            await manager.destroy_session("nonexistent-id")

    async def test_stop_closes_all_sessions(self) -> None:
        """stop() closes every active session."""
        manager = SessionManager()
        session_a = _make_session()
        session_b = _make_session()
        call_count = 0

        mock_sched = MagicMock()
        mock_sched.__aenter__ = AsyncMock(return_value=mock_sched)
        mock_sched.__aexit__ = AsyncMock(return_value=None)

        async def _make_session_side_effect(**_kwargs: object) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            return session_a if call_count == 1 else session_b

        mock_sched.session = AsyncMock(side_effect=_make_session_side_effect)

        with patch("exec_sandbox.api.manager.Scheduler", return_value=mock_sched):
            await manager.start()
            await manager.create_session("python", [])
            await manager.create_session("python", [])
            await manager.stop()

        session_a.close.assert_awaited_once()
        session_b.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# REST API tests
# ---------------------------------------------------------------------------


class TestRestCreateSandbox:
    """Tests for POST /api/v1/sandbox."""

    async def test_creates_sandbox_returns_session_id(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/sandbox", json={"language": "python", "packages": []})
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "test-session-id"

    async def test_default_language_is_python(self, client: AsyncClient, mock_manager: MagicMock) -> None:
        await client.post("/api/v1/sandbox", json={})
        mock_manager.create_session.assert_awaited_once_with(language="python", packages=[])

    async def test_packages_forwarded(self, client: AsyncClient, mock_manager: MagicMock) -> None:
        await client.post("/api/v1/sandbox", json={"language": "python", "packages": ["numpy"]})
        mock_manager.create_session.assert_awaited_once_with(language="python", packages=["numpy"])


class TestRestExecCode:
    """Tests for POST /api/v1/sandbox/{session_id}/exec."""

    async def test_exec_returns_stdout_stderr_exitcode(self, client: AsyncClient, mock_session: AsyncMock) -> None:
        mock_session.exec.return_value = _make_exec_result(stdout="hello\n", stderr="", exit_code=0)
        response = await client.post(
            "/api/v1/sandbox/test-session-id/exec",
            json={"code": "print('hello')"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "hello\n"
        assert data["stderr"] == ""
        assert data["exit_code"] == 0

    async def test_exec_nonzero_exit_code(self, client: AsyncClient, mock_session: AsyncMock) -> None:
        mock_session.exec.return_value = _make_exec_result(stderr="error", exit_code=1)
        response = await client.post(
            "/api/v1/sandbox/test-session-id/exec",
            json={"code": "raise SystemExit(1)"},
        )
        assert response.status_code == 200
        assert response.json()["exit_code"] == 1

    async def test_exec_unknown_session_returns_404(self, client: AsyncClient, mock_manager: MagicMock) -> None:
        mock_manager.get_session.side_effect = KeyError("unknown")
        response = await client.post(
            "/api/v1/sandbox/nonexistent/exec",
            json={"code": "pass"},
        )
        assert response.status_code == 404


class TestRestFileWrite:
    """Tests for POST /api/v1/sandbox/{session_id}/file/write."""

    async def test_write_file_success(self, client: AsyncClient, mock_session: AsyncMock) -> None:
        response = await client.post(
            "/api/v1/sandbox/test-session-id/file/write",
            json={"path": "hello.txt", "content": "hello world"},
        )
        assert response.status_code == 200
        assert response.json() == {"success": True}
        mock_session.write_file.assert_awaited_once_with("hello.txt", b"hello world")

    async def test_write_file_unknown_session_returns_404(self, client: AsyncClient, mock_manager: MagicMock) -> None:
        mock_manager.get_session.side_effect = KeyError("unknown")
        response = await client.post(
            "/api/v1/sandbox/nonexistent/file/write",
            json={"path": "x.txt", "content": "data"},
        )
        assert response.status_code == 404


class TestRestFileRead:
    """Tests for GET /api/v1/sandbox/{session_id}/file/read."""

    async def test_read_file_returns_content(self, client: AsyncClient, mock_session: AsyncMock) -> None:
        async def _read(_path: str, *, destination: io.BytesIO) -> None:
            destination.write(b"file content")

        mock_session.read_file.side_effect = _read

        response = await client.get(
            "/api/v1/sandbox/test-session-id/file/read",
            params={"path": "hello.txt"},
        )
        assert response.status_code == 200
        assert response.json()["content"] == "file content"

    async def test_read_file_unknown_session_returns_404(self, client: AsyncClient, mock_manager: MagicMock) -> None:
        mock_manager.get_session.side_effect = KeyError("unknown")
        response = await client.get(
            "/api/v1/sandbox/nonexistent/file/read",
            params={"path": "x.txt"},
        )
        assert response.status_code == 404


class TestRestListFiles:
    """Tests for GET /api/v1/sandbox/{session_id}/files."""

    async def test_list_files_returns_entries(self, client: AsyncClient, mock_session: AsyncMock) -> None:
        mock_session.list_files.return_value = [
            FileInfo(name="main.py", is_dir=False, size=42),
            FileInfo(name="data", is_dir=True, size=0),
        ]
        response = await client.get("/api/v1/sandbox/test-session-id/files")
        assert response.status_code == 200
        items = response.json()
        assert len(items) == 2
        assert items[0] == {"name": "main.py", "is_dir": False, "size": 42}
        assert items[1] == {"name": "data", "is_dir": True, "size": 0}

    async def test_list_files_empty_root(self, client: AsyncClient, mock_session: AsyncMock) -> None:
        mock_session.list_files.return_value = []
        response = await client.get("/api/v1/sandbox/test-session-id/files")
        assert response.status_code == 200
        assert response.json() == []

    async def test_list_files_unknown_session_returns_404(self, client: AsyncClient, mock_manager: MagicMock) -> None:
        mock_manager.get_session.side_effect = KeyError("unknown")
        response = await client.get("/api/v1/sandbox/nonexistent/files")
        assert response.status_code == 404


class TestRestDestroySandbox:
    """Tests for DELETE /api/v1/sandbox/{session_id}."""

    async def test_destroy_returns_204(self, client: AsyncClient, mock_manager: MagicMock) -> None:
        response = await client.delete("/api/v1/sandbox/test-session-id")
        assert response.status_code == 204
        mock_manager.destroy_session.assert_awaited_once_with("test-session-id")

    async def test_destroy_unknown_session_returns_404(self, client: AsyncClient, mock_manager: MagicMock) -> None:
        mock_manager.destroy_session.side_effect = KeyError("unknown")
        response = await client.delete("/api/v1/sandbox/nonexistent")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# MCP tool dispatch unit tests (no HTTP layer)
# ---------------------------------------------------------------------------


class TestMcpToolDispatch:
    """Unit tests for the individual MCP tool handler functions."""

    async def test_sandbox_create(self, mock_manager: MagicMock) -> None:
        from exec_sandbox.api.routes_mcp import _TOOL_DISPATCH

        routes_mcp.set_manager(mock_manager)
        result = await _TOOL_DISPATCH["sandbox_create"]({"language": "python", "packages": []}, mock_manager)
        assert result == {"session_id": "test-session-id"}

    async def test_sandbox_exec(self, mock_manager: MagicMock, mock_session: AsyncMock) -> None:
        from exec_sandbox.api.routes_mcp import _TOOL_DISPATCH

        mock_session.exec.return_value = _make_exec_result(stdout="42\n")
        result = await _TOOL_DISPATCH["sandbox_exec"](
            {"session_id": "test-session-id", "code": "print(42)"},
            mock_manager,
        )
        assert result["stdout"] == "42\n"
        assert result["exit_code"] == 0

    async def test_sandbox_exec_unknown_session(self, mock_manager: MagicMock) -> None:
        from exec_sandbox.api.routes_mcp import _TOOL_DISPATCH

        mock_manager.get_session.side_effect = KeyError("unknown")
        result = await _TOOL_DISPATCH["sandbox_exec"](
            {"session_id": "bad", "code": "pass"},
            mock_manager,
        )
        assert "error" in result

    async def test_sandbox_write_file(self, mock_manager: MagicMock, mock_session: AsyncMock) -> None:
        from exec_sandbox.api.routes_mcp import _TOOL_DISPATCH

        result = await _TOOL_DISPATCH["sandbox_write_file"](
            {"session_id": "test-session-id", "path": "x.txt", "content": "data"},
            mock_manager,
        )
        assert result == {"success": True}
        mock_session.write_file.assert_awaited_once_with("x.txt", b"data")

    async def test_sandbox_read_file(self, mock_manager: MagicMock, mock_session: AsyncMock) -> None:
        from exec_sandbox.api.routes_mcp import _TOOL_DISPATCH

        async def _read(_path: str, *, destination: io.BytesIO) -> None:
            destination.write(b"content")

        mock_session.read_file.side_effect = _read
        result = await _TOOL_DISPATCH["sandbox_read_file"](
            {"session_id": "test-session-id", "path": "x.txt"},
            mock_manager,
        )
        assert result == {"content": "content"}

    async def test_sandbox_list_files(self, mock_manager: MagicMock, mock_session: AsyncMock) -> None:
        from exec_sandbox.api.routes_mcp import _TOOL_DISPATCH

        mock_session.list_files.return_value = [FileInfo(name="a.py", is_dir=False, size=10)]
        result = await _TOOL_DISPATCH["sandbox_list_files"](
            {"session_id": "test-session-id", "path": ""},
            mock_manager,
        )
        assert result["files"] == [{"name": "a.py", "is_dir": False, "size": 10}]

    async def test_sandbox_destroy(self, mock_manager: MagicMock) -> None:
        from exec_sandbox.api.routes_mcp import _TOOL_DISPATCH

        result = await _TOOL_DISPATCH["sandbox_destroy"](
            {"session_id": "test-session-id"},
            mock_manager,
        )
        assert result == {"success": True}
        mock_manager.destroy_session.assert_awaited_once_with("test-session-id")

    async def test_sandbox_destroy_unknown_session(self, mock_manager: MagicMock) -> None:
        from exec_sandbox.api.routes_mcp import _TOOL_DISPATCH

        mock_manager.destroy_session.side_effect = KeyError("unknown")
        result = await _TOOL_DISPATCH["sandbox_destroy"](
            {"session_id": "bad"},
            mock_manager,
        )
        assert "error" in result

    async def test_unknown_tool_returns_error(self, mock_manager: MagicMock) -> None:
        """Unknown tool names return an error dict."""
        from exec_sandbox.api.routes_mcp import call_tool_handler

        routes_mcp.set_manager(mock_manager)
        result = await call_tool_handler("nonexistent_tool", {})
        import json

        payload = json.loads(result[0].text)
        assert "error" in payload

    async def test_list_tools_returns_six_tools(self) -> None:
        """list_tools_handler returns exactly the 6 expected tools."""
        from exec_sandbox.api.routes_mcp import list_tools_handler

        tools = await list_tools_handler()
        names = {t.name for t in tools}
        assert names == {
            "sandbox_create",
            "sandbox_exec",
            "sandbox_write_file",
            "sandbox_read_file",
            "sandbox_list_files",
            "sandbox_destroy",
        }
