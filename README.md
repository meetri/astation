# astation

A research workspace for [Hermes Agent](https://github.com/NousResearch/hermes-agent), delivered as
a plugin.

Hermes gives you agents and a conversation. astation adds the durable layer around them: projects
that group work, sessions that survive restarts, a searchable transcript, artifacts with folders and
tags, speech in and out, and an audit trail of what each session actually did on the machine.

It runs inside the Hermes dashboard process. No second container, no extra port, no separate
credential — Hermes's own authentication is the only gate.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Verifying the install](#verifying-the-install)
- [API](#api)
- [The audit layer](#the-audit-layer)
- [Configuration](#configuration)
- [Upgrading](#upgrading)
- [Development](#development)

## Requirements

| | |
|---|---|
| Hermes Agent | 0.21 or later |
| Python | 3.11+ (uses the interpreter Hermes runs on) |
| Storage | SQLite, created on first start |

The audit layer is optional and has its own requirements. See
[`audit-setup/AUDIT_SETUP.md`](audit-setup/AUDIT_SETUP.md).

## Installation

Install the plugin, then the Python packages Hermes does not ship, then restart the dashboard once.

```bash
hermes plugins install meetri/astation --ref <commit-sha> --enable
```

```bash
uv pip install --python /opt/hermes/.venv/bin/python --target "$HERMES_HOME/lazy-packages" \
  "sqlalchemy>=2" alembic pydantic-settings "python-multipart>=0.0.32" \
  "piper-tts>=1.7.0" "edge-tts>=7.2.8"
```

Speech is optional. Without `piper-tts` and `edge-tts` the plugin starts normally and the speech
routes return 503; everything else is unaffected.

Two things commonly surprise people on a first install:

**Dependencies are not installed automatically** when a plugin directory is copied into place by
hand. Only `hermes plugins install` does that. Without them the plugin comes up with
`routers_mounted: 0` and an `import_error`, which the health endpoint reports verbatim.

**Routes mount once, at dashboard startup.** Every install and every upgrade needs one dashboard
restart. `/api/dashboard/plugins/rescan` reloads interface bundles only.

### Multiple agent profiles

Each Hermes profile is a separate home with its own configuration and its own plugin directory, so
the plugin is enabled per profile:

```bash
hermes -p <profile> plugins enable astation
```

Profiles that do not have it enabled are unaffected. Session attribution in the audit layer only
covers profiles where it is enabled.

## Verifying the install

```
GET /api/plugins/astation/health
```

A healthy instance reports `status: ok` and a non-zero `routers_mounted`. The same response carries
the migration state, whether background services started, and the error text if anything failed to
import, so a broken install explains itself rather than returning an empty page.

## API

Everything mounts under `/api/plugins/astation/api/`, behind Hermes's authentication. Roughly 110
endpoints across these groups:

| Group | What it covers |
|---|---|
| `projects` | Projects, membership, filing, instructions |
| `sessions` | Session lifecycle, resume, transcript, turns |
| `artifacts` | Files and outputs, folders, tags, collections, search |
| `runs` | Turn-level execution records |
| `audit` | Host and session audit queries |
| `prompts` | Approvals, clarifications, interrupts |
| `speak`, `transcribe` | Text to speech, speech to text |
| `sandbox` | Workspace file access |
| `config`, `instance` | Runtime settings and instance state |

A WebSocket at `/api/plugins/astation/ws/events` streams session and run events.

## The audit layer

astation can record what each session did, and answer questions about it afterwards.

Two independent recorders, joined only when read:

- **The agent's account.** Every tool call a session makes, tagged with the session, turn and call
  identifiers, with how long it took and whether it succeeded.
- **The host's record.** A kernel sensor observes processes, file writes and network connections,
  knowing nothing about sessions.

A session timeline presents both in labelled tiers and never merges them. This matters: the host's
record is produced outside the agent and cannot be forged by it, so the two disagreeing is itself
a signal. A third, weaker tier attributes processes that ran inside a turn on the same profile, and
is withheld when another session of that profile was running at the same time.

**Tool output is not recorded.** A file read returns the file, and the archive is append-only, so
anything written to it cannot later be removed. What is stored is the duration, the status and the
error class.

```
GET  /api/plugins/astation/api/audit/health
GET  /api/plugins/astation/api/audit/sessions/{id}/timeline
GET  /api/plugins/astation/api/audit/files
GET  /api/plugins/astation/api/audit/net
POST /api/plugins/astation/api/audit/query
GET  /api/plugins/astation/api/audit/schema
```

`POST /audit/query` accepts one read-only `SELECT`, capped and recorded. Read-only is enforced by
the database role and again by a validator in front of it.

The sensor, shipper and store live in [`audit-setup/`](audit-setup/) as a Compose project with an
installer and a self-test. A local-only deployment needs no cloud account.

**The plugin runs without any of it.** With no audit stack configured, session attribution is
disabled with a log line saying so, and the audit routes return 503 with a reason rather than
reporting that nothing happened.

## Configuration

Settings are read from the environment. All are optional; the defaults suit a single-host install.

| Variable | Purpose |
|---|---|
| `AUDIT_INGEST_URL` | Where session audit rows are sent. Unset disables attribution. |
| `AUDIT_CLICKHOUSE_URL` | Audit store, for the query routes. Unset returns 503. |
| `AUDIT_CLICKHOUSE_USER`, `AUDIT_CLICKHOUSE_PASSWORD` | Read-only store credentials. |
| `AUDIT_HOST_LABEL` | Name this host reports itself as. |

The `llm.profile_override` capability lets the plugin run a completion on a named profile's own
model. It is declared in the manifest and still has to be granted:

```yaml
plugins:
  entries:
    astation:
      llm:
        allow_profile_override: true
```

Without it, features that rewrite text on a profile whose provider has no directly reachable
endpoint return 503 instead of falling back to a different model.

## Upgrading

```bash
hermes plugins install meetri/astation --ref <new-sha>
```

Restart the dashboard afterwards. Database migrations run on mount, after the existing database is
copied aside.

## Development

This repository is generated and published as an installable unit; see
[`GENERATED.md`](GENERATED.md) for what that means for pull requests. Issues and discussion are
welcome here.

```bash
pytest tests/
```

## License

No license has been declared yet. Until one is added, all rights are reserved.
