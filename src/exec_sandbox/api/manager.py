"""Session manager for the exec-sandbox API.

Manages the lifecycle of the global Scheduler and all active Sessions.
Sessions are stored in an in-memory dictionary keyed by UUID4 strings.

Thread-safety: All operations are async and designed to be called from a
single asyncio event loop. Concurrent requests are handled by FastAPI's
async request handlers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from exec_sandbox.scheduler import Scheduler

if TYPE_CHECKING:
    from exec_sandbox.config import SchedulerConfig
    from exec_sandbox.session import Session

logger = logging.getLogger(__name__)


class SessionManager:
    """Global manager for the Scheduler and active Session instances.

    Lifecycle:
        manager = SessionManager()
        await manager.start()   # called during app startup
        ...                     # handle requests
        await manager.stop()    # called during app shutdown

    Sessions are keyed by UUID4 strings and stored in memory.
    All active sessions are closed on stop().
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        """Initialise with an optional SchedulerConfig.

        Args:
            config: Optional scheduler configuration. Uses library defaults if None.
        """
        self._config = config
        self._scheduler: Scheduler | None = None
        self._sessions: dict[str, Session] = {}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the scheduler.

        Must be called before any session operations.
        """
        self._scheduler = Scheduler(self._config)
        await self._scheduler.__aenter__()
        logger.info("SessionManager started")

    async def stop(self) -> None:
        """Stop all active sessions and the scheduler.

        Idempotent: safe to call even if start() was never called.
        """
        # Close all active sessions first
        for session_id, session in list(self._sessions.items()):
            try:
                await session.close()
            except Exception as exc:
                logger.exception("Error closing session %s during shutdown: %s", session_id, exc)
        self._sessions.clear()

        if self._scheduler is not None:
            await self._scheduler.__aexit__(None, None, None)
            self._scheduler = None

        logger.info("SessionManager stopped")

    # -------------------------------------------------------------------------
    # Session CRUD
    # -------------------------------------------------------------------------

    async def create_session(self, language: str, packages: list[str]) -> str:
        """Create a new sandbox session and return its UUID.

        Args:
            language: Programming language for the session (e.g. "python").
            packages: Packages to pre-install in the sandbox.

        Returns:
            UUID4 string identifying the new session.

        Raises:
            RuntimeError: If the manager has not been started.
        """
        if self._scheduler is None:
            raise RuntimeError("SessionManager not started")

        pkg_list: list[str] | None = packages if packages else None
        session = await self._scheduler.session(language=language, packages=pkg_list)
        session_id = uuid4().hex
        self._sessions[session_id] = session
        logger.info("Created session %s (language=%s)", session_id, language)
        return session_id

    def get_session(self, session_id: str) -> Session:
        """Return the Session for a given ID.

        Args:
            session_id: UUID4 string from create_session().

        Returns:
            The active Session.

        Raises:
            KeyError: If no session with that ID exists.
        """
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"Session '{session_id}' not found") from None

    async def destroy_session(self, session_id: str) -> None:
        """Close and remove the session with the given ID.

        Args:
            session_id: UUID4 string from create_session().

        Raises:
            KeyError: If no session with that ID exists.
        """
        session = self._sessions.pop(session_id)
        await session.close()
        logger.info("Destroyed session %s", session_id)
