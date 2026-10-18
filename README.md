# astation

A research workspace for [Hermes Agent](https://github.com/NousResearch/hermes-agent), as a plugin.

Hermes gives you agents and a conversation. astation adds the durable layer around them: projects
that group work, sessions that survive restarts, a searchable transcript, artifacts with folders
and tags, speech in and out, and an audit trail of what each session did on the machine.

It runs inside the Hermes dashboard process — no second container, no extra port, and no separate
credential. Hermes's own authentication is the only gate.

## Install

```bash
hermes plugins install meetri/astation --ref <commit-sha> --enable
```

Install the Python packages Hermes does not ship, then restart the dashboard once:

```bash
uv pip install --python /opt/hermes/.venv/bin/python --target "$HERMES_HOME/lazy-packages" \
  "sqlalchemy>=2" alembic pydantic-settings "python-multipart>=0.0.32" \
  "piper-tts>=1.7.0" "edge-tts>=7.2.7"
```

Speech is optional. Without `piper-tts` and `edge-tts` the plugin starts normally and the speech
endpoints return 503.

Each Hermes profile has its own configuration, so enable it per profile:

```bash
hermes -p <profile> plugins enable astation
```

Check it came up:

```
GET /api/plugins/astation/health
```

`status: ok` and a non-zero `routers_mounted` means it is running. The same response carries the
migration state and any import error, so a failed install explains itself.

## What it does

Everything mounts under `/api/plugins/astation/api/`, behind Hermes's authentication. Around 110
endpoints covering projects, sessions and transcripts, artifacts, runs, approvals, speech,
workspace files and audit. A WebSocket at `/api/plugins/astation/ws/events` streams session and
run events.

### The audit trail

astation records what each session did, and answers questions about it afterwards.

Two recorders, kept separate. The plugin records every tool call a session makes — the tool, its
arguments, how long it took, whether it worked — tagged with the session and turn. A kernel sensor
records the processes, file writes and network connections on the host, knowing nothing about
sessions. A session timeline shows both, labelled, and never merges them: the host's record is
produced outside the agent, so the two disagreeing is itself a signal.

Tool output is not recorded. A file read returns the file, and the archive is append-only, so
anything written to it cannot later be removed.

```
GET  /api/plugins/astation/api/audit/sessions/{id}/timeline
GET  /api/plugins/astation/api/audit/files
GET  /api/plugins/astation/api/audit/net
POST /api/plugins/astation/api/audit/query      # one capped, read-only SELECT
```

The sensor and store are in [`audit-setup/`](audit-setup/). The plugin runs without them: session
attribution is disabled with a log line saying so, and the audit endpoints return 503 with a
reason rather than an empty result.

## Configure

Read from the environment. All optional; the audit variables are only needed if you run the stack.

| Variable | Purpose |
|---|---|
| `AUDIT_INGEST_URL` | Where session audit rows are sent |
| `AUDIT_CLICKHOUSE_URL` | Audit store, for the query endpoints |
| `AUDIT_CLICKHOUSE_USER` / `_PASSWORD` | Read-only store credentials |
| `AUDIT_HOST_LABEL` | Name this host reports itself as |
| `RESEARCH_GATEWAY_RUN_EVENT_RETENTION_DAYS` | Default `90`. Tool inputs and outputs of finished runs older than this are replaced with a stub; `0` keeps everything |
| `RESEARCH_GATEWAY_RUN_EVENT_PRUNE_INTERVAL_S` | Default `86400`. How often the pruning pass runs |
| `ASTATION_CAPTURE_THREAD` | Default on. Set to `0` to run event persistence on the event loop again (the previous behaviour); only useful for diagnosing |

### Database size

Every tool call and result of a run is stored so the run's timeline can be replayed. That is the bulk
of the workspace database (about 78% on the reference deployment), so old runs are trimmed:
after `RESEARCH_GATEWAY_RUN_EVENT_RETENTION_DAYS` the payloads of tool events in *finished* runs are
replaced with a small stub. Rows, ordering, messages, usage statistics and approvals are untouched,
and a run that is still open is never trimmed. The file stops growing but does not shrink by itself;
to give the space back to the filesystem, stop Hermes and run `VACUUM` on the database once.

The database and the event inbox run in SQLite's WAL mode with `synchronous=NORMAL`: a crash of
Hermes loses nothing, while a power cut or operating-system crash can roll back the last few
commits (it cannot corrupt the file). Back the database up with SQLite's backup API (or
`sqlite3 research.db ".backup file"`), not by copying the file, because recent commits can still be
in the `-wal` file next to it.

The plugin declares one capability, `llm.profile_override`, which lets it run a completion on a
named profile's own model. Grant it in Hermes's config:

```yaml
plugins:
  entries:
    astation:
      llm:
        allow_profile_override: true
```

## Troubleshooting

**`routers_mounted: 0` and an `import_error`.** The Python dependencies are missing. Copying the
plugin directory into place by hand does not install them; only `hermes plugins install` does.

**New routes return 404 after an upgrade.** Routes mount once, at dashboard startup. Restart it.
`/api/dashboard/plugins/rescan` reloads interface bundles only.

**Audit endpoints return 503.** The store is not configured or not reachable. This is deliberately
distinct from an empty result — the response says which.

**A session's audit timeline is empty.** The plugin is probably not enabled on that session's
profile. Check `hermes -p <profile> plugins list`.

**Migrations.** They run on mount, after the existing database is copied aside.

## Contributing

This repository is published as an installable unit and is generated from a development
repository, so pull requests against it would be overwritten by the next release. Open an issue
and it will be applied upstream and republished.

```bash
pytest tests/
```

## License

[BSD Zero Clause (0BSD)](LICENSE): use, copy, modify and distribute it for any purpose, with or
without attribution.

The plugin's own code is the only thing covered by that license. Its dependencies keep theirs, and
two of the optional speech packages are copyleft: `piper-tts` is GPL-3.0-or-later and `edge-tts` is
LGPL-3.0. The plugin only imports them when present and does not bundle them, so installing the
plugin does not change its own license, but if you redistribute an image or bundle that includes
them you take on their terms. The other dependencies (SQLAlchemy, Alembic, pydantic-settings,
faster-whisper) are MIT and `python-multipart` is Apache-2.0.
