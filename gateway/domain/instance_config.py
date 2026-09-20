"""`InstanceConfigCache`: the two `hermes config get` values `GET /api/vitals` serves."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Any

from adapters.hermes import HermesAdapter
from domain.hermes_runtime import _with_reconnect
from domain.models import utcnow

logger = logging.getLogger(__name__)

HERMES_CONFIG_GET_ARGV: tuple[str, ...] = ("config", "get")
SESSIONS_AUTO_PRUNE_KEY = "sessions.auto_prune"
SESSIONS_RETENTION_DAYS_KEY = "sessions.retention_days"


class InstanceConfigCache:
    """The two `hermes config get` values `GET /api/vitals` serves without waiting."""

    REFRESH_INTERVAL_S = 600.0

    def __init__(self, *, refresh_interval_s: float | None = None) -> None:
        self.refresh_interval_s = (
            self.REFRESH_INTERVAL_S if refresh_interval_s is None else refresh_interval_s
        )
        self.sessions_auto_prune: bool | None = None
        self.sessions_retention_days: int | None = None
        self.refreshed_at: datetime | None = None
        self._adapter: Any = None
        self._task: asyncio.Task[None] | None = None


    def snapshot(self) -> dict[str, Any]:
        return {
            "sessions_auto_prune": self.sessions_auto_prune,
            "sessions_retention_days": self.sessions_retention_days,
        }


    @staticmethod
    def _output_of(result: Any) -> str | None:
        """The CLI's stdout for a clean run, else None."""
        payload = result if isinstance(result, dict) else {}
        code = payload.get("code")
        if payload.get("blocked") is True:
            return None
        if isinstance(code, bool) or not isinstance(code, int) or code != 0:
            return None
        output = payload.get("output")
        return output if isinstance(output, str) else None

    @classmethod
    def parse_bool(cls, result: Any) -> bool | None:
        output = cls._output_of(result)
        if output is None:
            return None
        text = output.strip()
        if text == "true":
            return True
        if text == "false":
            return False
        return None

    @classmethod
    def parse_int(cls, result: Any) -> int | None:
        output = cls._output_of(result)
        if output is None:
            return None
        text = output.strip()
        if not text or not (text.isdigit() or (text[0] == "-" and text[1:].isdigit())):
            return None
        return int(text)


    async def _config_get(self, app_state: Any, key: str) -> Any:
        adapter: HermesAdapter = app_state.hermes_adapter
        argv = [*HERMES_CONFIG_GET_ARGV, key]
        return await _with_reconnect(app_state, adapter, lambda: adapter.cli_exec(argv))

    async def refresh_once(self, app_state: Any) -> None:
        """One pass: both keys, each independently best-effort."""
        try:
            self.sessions_auto_prune = self.parse_bool(
                await self._config_get(app_state, SESSIONS_AUTO_PRUNE_KEY)
            )
        except Exception as exc:
            logger.info("config get %s failed: %s", SESSIONS_AUTO_PRUNE_KEY, exc)
            self.sessions_auto_prune = None
        try:
            self.sessions_retention_days = self.parse_int(
                await self._config_get(app_state, SESSIONS_RETENTION_DAYS_KEY)
            )
        except Exception as exc:
            logger.info("config get %s failed: %s", SESSIONS_RETENTION_DAYS_KEY, exc)
            self.sessions_retention_days = None
        self.refreshed_at = utcnow()

    async def _run(self, app_state: Any) -> None:
        while True:
            await self.refresh_once(app_state)
            await asyncio.sleep(self.refresh_interval_s)

    def _reset_for(self, adapter: Any) -> None:
        self._adapter = adapter
        self.sessions_auto_prune = None
        self.sessions_retention_days = None
        self.refreshed_at = None

    def ensure_running(self, app_state: Any) -> None:
        """Start (or restart, for a new adapter) the refresher. Never awaits."""
        adapter = getattr(app_state, "hermes_adapter", None)
        if adapter is not self._adapter:
            self._reset_for(adapter)
            if self._task is not None and not self._task.done():
                self._task.cancel()
            self._task = None
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run(app_state))

    def start(self, app_state: Any) -> None:
        self.ensure_running(app_state)

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


def _instance_config_cache(app_state: Any) -> InstanceConfigCache:
    """The one cache per app, created on first use (see `InstanceConfigCache`)."""
    cache = getattr(app_state, "instance_config_cache", None)
    if not isinstance(cache, InstanceConfigCache):
        cache = InstanceConfigCache()
        app_state.instance_config_cache = cache
    return cache
