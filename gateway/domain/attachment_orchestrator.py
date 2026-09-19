"""`AttachmentOrchestrator`: the asynchronous two-turn attach (P3-3).

Moved out of `api/attachments.py` (CLEANUP_PLAN step 3.5); the upload / poll /
list / serve routes stay there and re-export these names.

Hermes has NO upload endpoint -- `/api/files*` is GET-only (measured,
`OPTIONS` -> 405, PV "Phase 3 probe") and `image.attach` takes only a *path
already on the sandbox* (P3-0b). The only measured way user-picked bytes
reach a conversation is the P3-0c chain, verified end to end on the live
instance (PV "Phase 3 build probes"):

    1. the gateway stores the upload and serves it at an unguessable
       capability URL the sandbox host can reach over LAN;
    2. a PRIMING TURN asks the agent to `curl` it into the sandbox
       (a "run EXACTLY this command" terminal one-liner -- the measured
       low-latency turn shape);
    3. the gateway polls `GET /api/files/download` for the sandbox copy and
       verifies it byte-exact against the upload's own sha256;
    4. only then: `image.attach` for an image, or -- for a PDF/document --
       the verified sandbox path becomes the reference the app puts in the
       user's actual message.

That takes MINUTES (~2.5-3.5 min measured floor for the fetch turn; 15 min
worst case for this instance's turns), so the whole thing is asynchronous by
construction: `POST .../attachments` returns 202 the moment the bytes are
stored, the orchestration runs as a background task writing ledger-style
state onto the `attachments` row, and the app POLLS `GET
/api/attachments/{id}` to show an honest "preparing attachment" indicator.
There is deliberately no synchronous path and no spinner to block on -- the
P3-0c decision forbids building one, and a prompt turn must never sit in a
synchronous UI path.

## Honesty rules

Same ledger discipline as `background_tasks` (P2-1): the row is written
before anything is sent to Hermes; every failure is a recorded state with a
`detail`, never silence; a gateway restart mid-flow marks in-flight rows
``orphaned`` (outcome genuinely unknown -- the curl may well have landed;
re-upload to retry). `redirected`/`steered`/`queued` submit outcomes are
recorded verbatim (B-05o): a redirect means the fetch command was merged
into an in-flight turn and may never run, which the verify timeout reports
honestly rather than the gateway guessing.
"""

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

#: How long the orchestrator waits for the sandbox copy to verify byte-exact
#: before declaring failure. 15 minutes = the worst turn latency ever
#: measured on this instance (PV, the queued-blocking pathology); the
#: measured floor is ~2.5-3.5 min of model time-to-tool.
VERIFY_TIMEOUT_S = 15 * 60

#: Poll cadence for the verify loop. Each miss costs one cheap upstream 404;
#: each hit streams the file once for the sha256 comparison.
VERIFY_POLL_INTERVAL_S = 10.0

#: `--max-time` for the priming turn's curl. LAN transfer of <=25MB is
#: sub-second measured at 12KB and low seconds worst-case; 120s is generous.
_CURL_MAX_TIME_S = 120

#: Row states (`domain/models.py::Attachment` docstring).
STATE_UPLOADED = "uploaded"
STATE_PRIMING = "priming"
STATE_ATTACHING = "attaching"
STATE_ATTACHED = "attached"
STATE_FAILED = "failed"
STATE_ORPHANED = "orphaned"

#: Terminal states: the serve token is dead, the poller can stop.
TERMINAL_STATES = frozenset({STATE_ATTACHED, STATE_FAILED, STATE_ORPHANED})

#: Non-terminal states a previous process may have left behind.
_IN_FLIGHT_STATES = (STATE_UPLOADED, STATE_PRIMING, STATE_ATTACHING)


#: Chunk size when reading the inbound upload stream and the verify stream.
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


# ---------------------------------------------------------------------------
# The orchestrator: upload -> prime -> verify -> attach, as a background task
# ---------------------------------------------------------------------------


class AttachmentOrchestrator:
    """Drives one attachment row through the measured two-turn chain.

    Built by `api.main.lifespan` AFTER `app.state` holds the adapter, the
    live-handle cache, the connect lock, the sessionmaker, and the artifact
    store -- it reads all of them off the one `app_state` it is handed, so
    every Hermes call goes through exactly the same `_with_reconnect` /
    `_with_live_handle` machinery as the interactive routes (two id spaces,
    B-15/B-23 retry rules included).

    Nothing here may crash its caller: `_run` contains every failure as a
    `failed` row with a `detail`. Tasks are held in a set so they are not
    garbage-collected mid-flight (held-task-set pattern).
    """

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state
        self._tasks: set[asyncio.Task[None]] = set()
        # Injectable timings so tests do not sleep 10s per poll.
        self.verify_timeout_s: float = VERIFY_TIMEOUT_S
        #: See `set_direct_delivery`. None keeps the priming-turn path.
        self._direct_delivery: Callable[[str, str], tuple[bool, str]] | None = None
        self.poll_interval_s: float = VERIFY_POLL_INTERVAL_S

    # -- startup sweep -----------------------------------------------------

    def orphan_on_startup(self) -> int:
        """Mark rows a previous process left in flight as `orphaned`.

        The orchestration task died with the old process and nothing rebuilds
        it (unlike background tasks there is no completion event to rescue
        with); the sandbox curl may or may not have landed -- genuinely
        unknown, so the row says so. Best-effort: an unmigrated DB logs and
        moves on (same as `BackgroundLedger.orphan_on_startup`).
        """
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

    # -- task scheduling ---------------------------------------------------

    def start_attach(self, attachment_id: str) -> None:
        task = asyncio.create_task(self._run(attachment_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def set_direct_delivery(
        self, deliver: Callable[[str, str], tuple[bool, str]] | None
    ) -> None:
        """Install an in-process way to put the bytes in the sandbox.

        The sidecar could not write to the Hermes sandbox, so it asked the
        AGENT to fetch the file: a priming turn telling it to `curl` a
        capability URL, then polling `files_download` until a byte-exact copy
        appeared, with a 15-minute ceiling. That chain exists only because of
        the process boundary -- every part of it is a workaround.

        Running inside Hermes the sandbox is a local directory, so the bytes
        can simply be written. `deliver(sandbox_path, storage_key)` returns
        `(ok, detail)`. When it is installed the priming turn and the poll are
        both skipped; when it is not, the original path runs unchanged.
        """
        self._direct_delivery = deliver

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # -- row IO ------------------------------------------------------------

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

    # -- the flow ----------------------------------------------------------

    async def _run(self, attachment_id: str) -> None:
        try:
            await self._run_inner(attachment_id)
        except asyncio.CancelledError:
            # Process shutdown mid-flow: the startup sweep of the next
            # process will orphan the row; write the honest state now if we
            # still can.
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

        # 0. In-process: write the bytes straight into the sandbox and skip
        # both the priming turn and the poll. Nothing is asked of the agent,
        # nothing has to be reachable over the LAN, and the 15-minute verify
        # ceiling stops applying.
        if self._direct_delivery is not None:
            self._set_state(attachment_id, STATE_ATTACHING, "writing the file into the sandbox")
            ok, detail = self._direct_delivery(sandbox_path, row.storage_key)
            if not ok:
                self._set_state(attachment_id, STATE_FAILED, detail)
                return
            await self._attach_and_finish(
                attachment_id, adapter, cache, stored_id, sandbox_path, detail
            )
            return

        # 1. The priming turn.
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
        # B-05o honesty: only `streaming` means the fetch command gets its own
        # turn now. queued runs later (keep waiting); redirected/steered were
        # merged into an in-flight turn and MAY never execute -- the verify
        # timeout is what reports that outcome truthfully.
        self._set_state(
            attachment_id,
            STATE_PRIMING,
            f"fetch turn submitted (status: {submit_status}); waiting for the "
            "sandbox copy to verify byte-exact",
        )

        # 2. Poll-verify: the sandbox copy must equal the upload, sha256.
        verified, verify_detail = await self._poll_verify(
            adapter, sandbox_path, expected_checksum, expected_size
        )
        if not verified:
            self._set_state(attachment_id, STATE_FAILED, verify_detail)
            return

        await self._attach_and_finish(
            attachment_id, adapter, cache, stored_id, sandbox_path, verify_detail
        )

    async def _attach_and_finish(
        self,
        attachment_id: str,
        adapter: HermesAdapter,
        cache: Any,
        stored_id: str,
        sandbox_path: str,
        verify_detail: str,
    ) -> None:
        """Everything after the bytes are known to be in the sandbox.

        Shared by the priming-turn path and the in-process one, so the two
        cannot drift in what they record or how they report a failure. A
        document is done once the file is there; an image still needs
        `image.attach`, which is an RPC either way.
        """
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
        """Blocking poll with an explicit deadline (never an idle wait).

        Each round fetches the sandbox path via the measured download route
        and compares sha256 against the upload's. A 404 (curl not landed
        yet), a size/checksum mismatch (partial write mid-curl), and a
        transient transport error all mean "poll again"; only the deadline
        ends it. Returns `(verified, human_detail)`.
        """
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
                                # Bigger than the upload plus slack: not our
                                # file; stop reading, report, keep polling.
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
