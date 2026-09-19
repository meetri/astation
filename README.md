# Agent Station

**`astation`** — a research workspace that runs inside Hermes.

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that adds a durable research
workspace: projects, sessions, runs, artifacts, chat history, speech, and a phone-friendly API for
the trg-research iOS app.

It runs **inside** the Hermes dashboard process. There is no second container, no Docker socket,
and no separate credential — Hermes's own auth gate is the only one.

## Install

```bash
# in the Hermes container / on the Hermes host
hermes plugins install <owner>/hermes-astation --ref <commit-sha> --enable
```

Then install the Python packages Hermes does not already ship, and restart the dashboard:

```bash
uv pip install --python /opt/hermes/.venv/bin/python --target "$HERMES_HOME/lazy-packages" \
  "sqlalchemy>=2" alembic pydantic-settings "python-multipart>=0.0.32" \
  "piper-tts>=1.7.0" "edge-tts>=7.2.8"
```

**Dependencies are not installed automatically** when a plugin directory is simply dropped in —
only a real `hermes plugins install` does that, and even then the speech extras are optional. If
they are missing the plugin comes up with `routers_mounted: 0` and an `import_error`, which
`GET /health` reports verbatim.

**Plugin API routes mount once at dashboard startup**, so every install or upgrade needs one
dashboard restart. `/api/dashboard/plugins/rescan` reloads UI bundles only.

## Check it

```
GET /api/plugins/astation/health
```

A healthy instance reports `status: ok`, `routers_mounted: 23`, `services_started: true`,
`foreign_prompts: "hook"` and `ws_core_module: "hermes_cli.web_server_chat"` (the last is
version-sensitive; on Hermes 0.20.5 the same helper lives in `hermes_cli.web_server`).

It also reports `import_error`, `startup_error` and `migration` verbatim — read it first when
something is wrong, because a plugin that fails to mount is otherwise only one line in a log.

## What it needs

- **Hermes 0.21.3 or later.** It uses `session.create`'s `profile` parameter, which earlier
  versions silently ignore, and plugin hooks for capture.
- **Its own data directory**, `$HERMES_HOME/astation/`: the SQLite workspace database, the
  content-addressed artifact store, Piper voices and the Whisper model cache. Inside the Hermes
  data volume, so it survives an image rebuild.

## How it authenticates

Everything is behind Hermes's dashboard auth gate. Clients log in with
`POST /auth/password-login` and keep the session cookie; a WebSocket upgrade takes a single-use
`?ticket=` from `POST /api/auth/ws-ticket`.

Note that Hermes's HTTP auth gates do **not** run for WebSocket routes — Starlette does not
invoke HTTP middleware for a websocket scope. This plugin's `/ws/events` therefore calls Hermes's
own `_ws_auth_ok` and `_ws_request_is_allowed` itself and **fails closed**.

## Migrations

The workspace schema is migrated on load, after copying the database aside. A migration failure
refuses to serve rather than serving a half-migrated database. `hermes trg migrate` runs it by
hand.

## This directory is generated

Edit `plugin/` in the `astation` repo and re-run `scripts/build_plugin_repo.sh`. See
`GENERATED.md` for the source commit.
