# audit-setup

The host half of astation's audit trail: a kernel sensor, a shipper and a queryable store, as a
Compose project.

It records what happens on the machine — every process executed, every file created, written,
renamed or deleted, and every TCP connection — and makes it queryable. The plugin ties that
activity to the agent session that caused it.

Three containers:

| | | |
|---|---|---|
| **Tetragon** | kernel sensor | Watches syscalls through eBPF. Privileged; sees the whole host. |
| **Vector** | shipper | Parses, redacts secrets, routes by event class. |
| **ClickHouse** | store | One table per event class, each with its own retention. |

## Setup

```bash
cp .env.example .env          # set AUDIT_HOST_LABEL; the rest have defaults
./install.sh --preflight      # check this host can run it; changes nothing
./install.sh                  # install, then self-test
```

That is a complete working stack with no cloud account. Passwords are generated on first run and
written back into `.env`.

The installer is idempotent. Change any config or `.env` value and run it again; it re-renders,
recreates what changed, and re-runs the self-test.

```bash
./install.sh --no-test        # skip the self-test
./install.sh --uninstall      # stop and remove containers, keep the data
./selftest.sh                 # 21 checks against a running stack
```

### Connect the plugin

```bash
AUDIT_INGEST_URL=http://<shipper-host>:8686/ingest
AUDIT_CLICKHOUSE_URL=http://<store-host>:8123
AUDIT_CLICKHOUSE_USER=reader
AUDIT_CLICKHOUSE_PASSWORD=<the reader password from .env>
AUDIT_HOST_LABEL=<this host's label>
```

Both must be reachable from wherever the agent runs.

### Archiving

By default nothing leaves the machine, which means the record is not tamper-evident: anyone with
root can edit the store. Set `AUDIT_BUCKET` and AWS credentials to also write to S3 under Object
Lock, where objects cannot be altered or deleted for their retention period — including by you.

Turning it on later is not a migration. Set the variables and re-run `./install.sh`.

## Retention

`retention.yaml` is the only place retention is decided. It sets how long each class of event
stays queryable, and how long archived copies are kept. Edit it and re-run `./install.sh`.

Defaults run from 7 days for the liveness heartbeat to 90 days for authentication events. File
events get the shortest window because they are by far the highest volume — a build writes
millions of them.

## Access

Three database roles, each doing one job:

| Role | Can |
|---|---|
| `ingest` | Insert only. Cannot read what it wrote. |
| `reader` | Select only, forced read-only, with row and memory limits. |
| `admin` | Schema changes. Loopback only. |

The plugin uses `reader`. Every query it runs is itself recorded.

## Troubleshooting

**The sensor will not start.** It needs a kernel with BTF (`/sys/kernel/btf/vmlinux`) and
`CAP_SYS_ADMIN`. `./install.sh --preflight` checks both and changes nothing.

**A policy change kills the sensor.** Tetragon refuses more than four values in a selector, and a
malformed policy takes down the whole daemon rather than just that policy. Read the header of
`tetragon/policies/file-mutations.yaml` before editing — the selectors are narrower than they look.

**The shipper starts but nothing lands.** Its config is rendered from a template, because Vector
does not read environment variables in config values at all. Edit `vector/vector.yaml`, not the
rendered copy, and re-run `./install.sh`.

**Config changes appear to do nothing.** A running container's bind mounts stay attached to the
old file. The installer forces container recreation for this reason; if you ran `docker compose
up` by hand, add `--force-recreate`.

**The store will not start after editing users.** A `--` inside an XML comment is a parse error,
and `<grants>` alongside `<access_management>` on one user causes a restart loop.

**Checking it works.** `./selftest.sh` performs a real action for each check and then looks for
its record: an in-container process carrying a container id, a file created, renamed and deleted,
inbound and outbound connections, a secret masked rather than dropped, the reader role refused a
write, and every table carrying a retention policy.

## Layout

| Path | |
|---|---|
| `docker-compose.yml` | The three services. Images pinned by digest. |
| `install.sh` | Renders config, starts the stack, runs the self-test |
| `selftest.sh` | 21 checks against a running stack |
| `.env.example` | Every setting, marked required or optional |
| `retention.yaml` | Retention policy, per event class |
| `tetragon/policies/` | What the kernel sensor watches |
| `vector/vector.yaml` | Parsing, redaction, routing. A template |
| `clickhouse/` | Schema, roles and server settings |
