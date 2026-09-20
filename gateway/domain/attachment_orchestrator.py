"""`AttachmentOrchestrator`: the asynchronous two-turn attach (P3-3)."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import posixpath
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy import select

from adapters.hermes import HermesAdapter, HermesError
from domain.hermes_runtime import _with_live_handle, _with_reconnect
from domain.models import Attachment

logger = logging.getLogger(__name__)

VERIFY_TIMEOUT_S = 15 * 60

VERIFY_POLL_INTERVAL_S = 10.0

_CURL_MAX_TIME_S = 120

STATE_UPLOADED = "uploaded"
STATE_PRIMING = "priming"
STATE_ATTACHING = "attaching"
STATE_ATTACHED = "attached"
STATE_FAILED = "failed"
STATE_ORPHANED = "orphaned"

TERMINAL_STATES = frozenset({STATE_ATTACHED, STATE_FAILED, STATE_ORPHANED})

_IN_FLIGHT_STATES = (STATE_UPLOADED, STATE_PRIMING, STATE_ATTACHING)


_UPLOAD_CHUNK_BYTES = 64 * 1024


def build_prime_command(sandbox_path: str, serve_url: str) -> str:
    """The priming turn's one-liner. Everything interpolated here is
    gateway-constructed: the path is attachment-dir + row id + sanitized
    filename, and the URL is base + token. Nothing user-controlled can break
    out of the single quotes because `sanitize_filename` leaves no quote,
    space, or metacharacter to break out with."""
    directory = posixpath.dirname(sandbox_path)
    return (
        f"mkdir -p '{directory}' && "
        f"curl -sS --fail --max-time {_CURL_MAX_TIME_S} "
        f"-o '{sandbox_path}' '{serve_url}' && "
        f"sha256sum '{sandbox_path}'"
    )


def build_prime_text(sandbox_path: str, serve_url: str) -> str:
    """The full priming prompt. "Run EXACTLY this command" is the measured
    low-latency turn shape (the P3-0c/P3-0d probes' own discipline: terminal
    one-liners, no model reasoning asked for)."""
    return (
        "Run EXACTLY this one terminal command and reply with only its "
        "output, nothing else:\n"
        f"{build_prime_command(sandbox_path, serve_url)}"
    )


class AttachmentOrchestrator:
    """Drives one attachment row through the measured two-turn chain."""

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state
        self._tasks: set[asyncio.Task[None]] = set()
        self.verify_timeout_s: float = VERIFY_TIMEOUT_S
        self._direct_delivery: Callable[[str, str], tuple[bool, str]] | None = None
        self.poll_interval_s: float = VERIFY_POLL_INTERVAL_S


    def orphan_on_startup(self) -> int:
        """Mark rows a previous process left in flight as `orphaned`."""
        factory = getattr(self._app_state, "db_sessions", None)
        if factory is None:  # pragma: no cover - defensive
            return 0
        try:
            with factory() as db:
                rows = list(
                    db.execute(
                        select(Attachment).where(Attachment.state.in_(_IN_FLIGHT_STATES))
                    ).scalars()
                )
                for row in rows:
                    row.state = STATE_ORPHANED
                    row.detail = (
                        "the gateway restarted while this attachment was being "
                        "prepared; whether the sandbox copy landed is unknown -- "
                        "re-attach the file to retry"
                    )
                db.commit()
                if rows:
                    logger.warning(
                        "orphaned %d in-flight attachment(s) from a previous gateway process",
                        len(rows),
                    )
                return len(rows)
        except Exception:
            logger.info(
                "attachments table not readable at startup (migration not run "
                "yet?); skipping the attachment orphan sweep"
            )
            return 0


    def start_attach(self, attachment_id: str) -> None:
        task = asyncio.create_task(self._run(attachment_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def set_direct_delivery(
        self, deliver: Callable[[str, str], tuple[bool, str]] | None
    ) -> None:
        """Install an in-process way to put the bytes in the sandbox."""
        self._direct_delivery = deliver

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


    def _load(self, attachment_id: str) -> Attachment | None:
        with self._app_state.db_sessions() as db:
            return db.get(Attachment, attachment_id)

    def _set_state(
        self,
        attachment_id: str,
        state: str,
        detail: str | None,
        *,
        attach_result: Any = None,
    ) -> None:
        with self._app_state.db_sessions() as db:
            row = db.get(Attachment, attachment_id)
            if row is None:  # pragma: no cover - row deleted underneath us
                return
            row.state = state
            row.detail = detail
            if attach_result is not None:
                row.attach_result_json = attach_result
            db.commit()


    async def _run(self, attachment_id: str) -> None:
        try:
            await self._run_inner(attachment_id)
        except asyncio.CancelledError:
            self._set_state(
                attachment_id,
                STATE_ORPHANED,
                "the gateway shut down while this attachment was being prepared",
            )
            raise
        except Exception as exc:
            logger.exception("attachment %s orchestration crashed", attachment_id)
            self._set_state(
                attachment_id,
                STATE_FAILED,
                f"internal error while preparing the attachment: {exc.__class__.__name__}: {exc}",
            )

    async def _run_inner(self, attachment_id: str) -> None:
        row = self._load(attachment_id)
        if row is None:  # pragma: no cover - deleted underneath us
            return
        adapter: HermesAdapter = self._app_state.hermes_adapter
        cache = self._app_state.live_handle_cache
        stored_id = row.stored_session_id
        profile = row.profile
        sandbox_path = row.sandbox_path
        expected_checksum = row.checksum
        expected_size = row.size_bytes
        serve_url = (
            row.attach_result_json.get("serve_url")
            if isinstance(row.attach_result_json, dict)
            else None
        )
        if not serve_url:  # pragma: no cover - route always records it
            self._set_state(attachment_id, STATE_FAILED, "no serve URL was recorded")
            return

        if self._direct_delivery is not None:
            self._set_state(attachment_id, STATE_ATTACHING, "writing the file into the sandbox")
            ok, detail = self._direct_delivery(sandbox_path, row.storage_key)
            if not ok:
                self._set_state(attachment_id, STATE_FAILED, detail)
                return
            await self._attach_and_finish(
                attachment_id, adapter, cache, stored_id, profile, sandbox_path, detail
            )
            return

        prime_text = build_prime_text(sandbox_path, serve_url)
        self._set_state(attachment_id, STATE_PRIMING, "asking the agent to fetch the file")
        try:
            _live, ack = await _with_reconnect(
                self._app_state,
                adapter,
                lambda: _with_live_handle(
                    adapter,
                    cache,
                    stored_id,
                    lambda live: adapter.prompt_submit(live, prime_text),
                    profile=profile,
                ),
            )
        except HermesError as exc:
            self._set_state(
                attachment_id,
                STATE_FAILED,
                f"could not submit the fetch turn to the session: {exc}",
            )
            return
        submit_status = (ack.get("status") if isinstance(ack, dict) else None) or "unknown"
        self._set_state(
            attachment_id,
            STATE_PRIMING,
            f"fetch turn submitted (status: {submit_status}); waiting for the "
            "sandbox copy to verify byte-exact",
        )

        verified, verify_detail = await self._poll_verify(
            adapter, sandbox_path, expected_checksum, expected_size
        )
        if not verified:
            self._set_state(attachment_id, STATE_FAILED, verify_detail)
            return

        await self._attach_and_finish(
            attachment_id, adapter, cache, stored_id, profile, sandbox_path, verify_detail
        )

    async def _attach_and_finish(
        self,
        attachment_id: str,
        adapter: HermesAdapter,
        cache: Any,
        stored_id: str,
        profile: str,
        sandbox_path: str,
        verify_detail: str,
    ) -> None:
        """Everything after the bytes are known to be in the sandbox."""
        row = self._load(attachment_id)
        if row is None:  # pragma: no cover - deleted underneath us
            return
        serve_url = (
            row.attach_result_json.get("serve_url")
            if isinstance(row.attach_result_json, dict)
            else None
        )
        if row.kind != "image":
            self._set_state(
                attachment_id,
                STATE_ATTACHED,
                verify_detail,
                attach_result={"serve_url": serve_url, "verify": verify_detail},
            )
            return

        self._set_state(attachment_id, STATE_ATTACHING, f"{verify_detail}; issuing image.attach")
        try:
            _live, result = await _with_reconnect(
                self._app_state,
                adapter,
                lambda: _with_live_handle(
                    adapter,
                    cache,
                    stored_id,
                    lambda live: adapter.request(
                        "image.attach", {"session_id": live, "path": sandbox_path}
                    ),
                    profile=profile,
                ),
            )
        except HermesError as exc:
            self._set_state(
                attachment_id,
                STATE_FAILED,
                f"the file is in the sandbox at {sandbox_path!r} but "
                f"image.attach failed: {exc}",
            )
            return
        if not (isinstance(result, dict) and result.get("attached") is True):
            self._set_state(
                attachment_id,
                STATE_FAILED,
                f"image.attach answered without attached=true: {result!r}",
                attach_result={"serve_url": serve_url, "image_attach": result},
            )
            return
        self._set_state(
            attachment_id,
            STATE_ATTACHED,
            f"{verify_detail}; image.attach confirmed "
            f"({result.get('width')}x{result.get('height')}, "
            f"~{result.get('token_estimate')} tokens)",
            attach_result={"serve_url": serve_url, "image_attach": result},
        )

    async def _poll_verify(
        self,
        adapter: HermesAdapter,
        sandbox_path: str,
        expected_checksum: str,
        expected_size: int,
    ) -> tuple[bool, str]:
        """Blocking poll with an explicit deadline (never an idle wait)."""
        deadline = time.monotonic() + self.verify_timeout_s
        last_observation = "the sandbox copy never appeared"
        polls = 0
        while time.monotonic() < deadline:
            polls += 1
            try:
                response = await adapter.files_download(sandbox_path)
            except HermesError as exc:
                last_observation = f"download probe failed: {exc}"
            else:
                if response.status_code != 200:
                    await response.aclose()
                    last_observation = (
                        f"sandbox copy not fetchable yet (HTTP {response.status_code})"
                    )
                else:
                    digest = hashlib.sha256()
                    size = 0
                    stream_error: str | None = None
                    try:
                        async for chunk in response.aiter_bytes():
                            digest.update(chunk)
                            size += len(chunk)
                            if size > expected_size + _UPLOAD_CHUNK_BYTES:
                                stream_error = (
                                    f"sandbox copy is larger than the upload "
                                    f"({size}+ vs {expected_size} bytes)"
                                )
                                break
                    except Exception as exc:
                        stream_error = f"verify stream failed: {exc}"
                    finally:
                        await response.aclose()
                    if stream_error is None:
                        checksum = digest.hexdigest()
                        if checksum == expected_checksum and size == expected_size:
                            return True, (
                                f"sandbox copy verified byte-exact "
                                f"(sha256 match, {size} bytes, poll {polls})"
                            )
                        last_observation = (
                            f"sandbox copy present but not yet byte-exact "
                            f"({size} of {expected_size} bytes)"
                        )
                    else:
                        last_observation = stream_error
            await asyncio.sleep(self.poll_interval_s)
        return False, (
            f"the sandbox copy did not verify byte-exact within "
            f"{int(self.verify_timeout_s)}s ({last_observation}). The fetch "
            "turn may not have run (a busy session queues or redirects turns) "
            "-- re-attach to retry"
        )
