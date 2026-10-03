#!/usr/bin/env bash
# Exit non-zero when any Hermes session is mid-turn: gates a restart, which drops every live
# session and every profile gateway. Uses the gateway's own adapter and the HERMES_* values
# in the gitignored .env.
set -euo pipefail
cd "$(dirname "$0")/.."
envval() { sed -n "s/^$1=//p" .env | tail -1; }
for name in HERMES_SCHEME HERMES_HOST HERMES_PORT HERMES_USERNAME HERMES_PASSWORD; do
  export "$name=$(envval "$name")"
done
cd gateway
exec uv run --quiet --no-project --with httpx --with websockets --with pydantic-settings \
  python - <<'PY'
import asyncio
import sys

sys.path.insert(0, ".")
from adapters.hermes.client import HermesAdapter


async def main() -> int:
    async with HermesAdapter() as adapter:
        await adapter.login()
        await adapter.connect()
        rows = (await adapter.request("session.active_list", {})).get("sessions") or []
        busy = [s for s in rows if s.get("status") not in ("idle", None)]
        print(f"active: {len(rows)} | busy: {len(busy)}")
        for s in busy:
            print("   BUSY:", s.get("status"), "|", (s.get("title") or "")[:50])
        return 1 if busy else 0


sys.exit(asyncio.run(main()))
PY
