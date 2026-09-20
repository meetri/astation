# AUDIT_SETUP — the full stack, explained

This directory is a self-contained Docker stack that records what happens on a host at the
kernel — every process executed with its full argv, every file created, written, renamed or
deleted, every TCP connection made or accepted — plus what the agent layer above it did, and
makes all of it queryable as SQL. It is append-only: rows are written, never edited, and
(optionally) copied within a minute to an S3 bucket the host itself cannot alter or delete.
The astation plugin in this repo reads that store to answer "what did this session actually
do on the machine".

You do not need to have read anything else. `README.md` next to this file is the short
quick start; this file is the explanation.

**Quickest possible start:** `AUDIT_HOST_LABEL=some-name` in `.env`, then `./install.sh`.
That is a complete working stack with no AWS account — see [§8](#8-local-only-vs-archived).

---

## 1. The three images

The stack is three containers and nothing else. Each image is pinned **by digest** in
`docker-compose.yml`, not by tag: a tag can be moved, a digest cannot, and a sensor that
silently changed version after a `pull` is a sensor whose output shape you no longer know.

### `quay.io/cilium/tetragon:v1.7.1` — the sensor

**What it does.** Loads eBPF programs into the running kernel and writes one JSON object per
observed event to a file. Two tracing policies are mounted from `tetragon/policies/`:

| Policy | Hooks | Captures |
|---|---|---|
| `file-mutations.yaml` | `sys_openat` (write-flag mask), `sys_unlinkat`, `sys_renameat2`, rate-limited `security_file_permission` | every file written, deleted, renamed |
| `network.yaml` | `tcp_connect`, `tcp_set_state`, `tcp_close` | outbound and inbound TCP, with the owning process |

There is no `exec.yaml`: Tetragon exports process exec and exit events without a tracing policy,
so `host_exec` and `host_exit` come from the daemon's own output.

**Why it, and not the alternatives.** auditd covers exec well, but whole-filesystem write rules
are expensive on it and it has no container awareness; Tetragon stamps every event with the
cgroup-derived container id, which is what ties an event to the container that caused it.
Falco and osquery were not on the host either; the box was greenfield. auditd remains available as an independent second exec
witness if a later review wants one.

**What it needs from the host.** This is the most powerful container on the box, and all three
of these are load-bearing rather than incidental:

- `privileged: true` — eBPF programs attach to kernel functions.
- `pid: host` — the process tree must be the host's tree, not the container's.
- `cgroup: host` — cgroup identity is how an event is attributed to a container.
- `/sys/kernel/btf/vmlinux:ro` — BTF is what lets the probes load with no kernel headers and no
  build step. **Its absence is the one hard stop**; `install.sh` refuses to continue without it.
- `${AUDIT_DATA_DIR}/tetragon-log` — where the JSON export is written (10 MB × 5 backups,
  mode 0644).
- `./tetragon/policies:ro` — the policies.

No ports. It talks to nothing; it writes a file.

**What breaks without it.** Everything host-level. `host_exec`, `host_exit`, `host_file` and
`host_net` stop, and since the only thing that writes them is the export file, the store simply
stops growing. `host_auth` (journald) and the agent classes would keep flowing, which is the
dangerous shape: the pipeline looks alive while the kernel record is gone. That is exactly what
`selftest.sh` exists to catch.

### `timberio/vector:0.58.0-debian` — the shipper

**What it does.** One process that reads every source on the host, normalises it into the
schema, redacts secrets, routes by event class, and fans out to the sinks. Sources:

| Source | Type | What it is |
|---|---|---|
| `tetragon` | `file` | the export file above, `read_from: beginning` with a checkpoint |
| `journal` | `journald` | `ssh`, `sshd`, `sudo`, `docker` units — logins and privilege changes |
| `beat` | `demo_logs` | one synthetic liveness event a minute |
| `agent_http` | `http_server` on `0.0.0.0:8686`, path `/ingest` | the plugin's `agent_*` rows |

**Why it, and not the alternatives.** One static binary covering file tailing, journald, an HTTP
listener, a transform language and both sinks — no separate agent per source. Its disk buffers
with `when_full: block` are the property that matters here: a ClickHouse restart or a network
blip must never drop audit data, so the shipper waits rather than discards.

**What it needs from the host.**

- `./vector/vector.rendered.yaml:ro` — the **rendered** config, not the template (see [§5](#5-how-the-installer-works)).
- `${AUDIT_DATA_DIR}/tetragon-log:ro` — the export file.
- `${AUDIT_DATA_DIR}/vector` — the file checkpoint and the disk buffers, on a host volume so a
  container replacement resumes instead of re-reading or skipping.
- `/var/log/journal:ro`, `/run/log/journal:ro`, `/etc/machine-id:ro` — the journald source.
- `user: "0:0"` — Tetragon writes the export as root; Vector has to be able to read it.
- Port 8686 is bound inside the container and published to **no** host interface. It is
  reachable only by name (`http://audit-vector:8686/ingest`) from containers on the `audit`
  docker network.

**What breaks without it.** All of it. Tetragon's export file rotates through 5 × 10 MB and then
deletes the oldest — so with no shipper, events are destroyed unread within minutes on a busy
host. Nothing reaches ClickHouse, nothing reaches S3, and the liveness beat stops, which is what
the off-box heartbeat alarms on.

### `clickhouse/clickhouse-server:26.8.8.8` — the hot store

**What it does.** The queryable store. One MergeTree table per event class
(`clickhouse/schema.sql`), all `PARTITION BY toDate(ts)` and `ORDER BY (host, ts)`, with
`LowCardinality` on the small-vocabulary columns. Measured on a working install: **30.9 bytes per
row** on disk after compression.

Tables: `host_exec`, `host_exit`, `host_file`, `host_net`, `host_auth`, `host_container`,
`host_beat`, `agent_tool`, `agent_llm`, `agent_session`, `audit_query`.

**Why it, and not the alternatives.** Columnar SQL with 10–20× compression on this shape of data,
per-table TTLs, and a fixed schema an LLM analyst can write queries against.
Elasticsearch/OpenSearch was rejected on memory footprint and on query surface; Loki was rejected
because it indexes labels, not fields, and every question here is a field question ("every
process that wrote under this path from this session").

**What it needs from the host.**

- `user: "101:101"` — the image drops to uid 101, and `install.sh` chowns the data and log
  directories to match. A mismatch aborts startup with `MISMATCHING_USERS_FOR_PROCESS_AND_DATA`
  and says so only in a logfile inside the container.
- `${AUDIT_DATA_DIR}/clickhouse` — the data.
- `${AUDIT_DATA_DIR}/clickhouse-log` — mounted **out** on purpose: a container stuck in a restart
  loop takes its own logs with it, and the startup error is written here and nowhere else.
- `./clickhouse/users.d:ro`, `./clickhouse/config.d:ro`.
- `CLICKHOUSE_SKIP_USER_SETUP=1` — without it the entrypoint creates a passwordless `default`
  superuser. `users.d` also removes that account; this stops it being created at all.
- `ports: "127.0.0.1:${AUDIT_CLICKHOUSE_HTTP_PORT}:8123"` — **loopback only.** Reachable from
  other containers on the `audit` network by name, and from this host's own shell. It is not on
  the LAN; a sensor host reaches it over WireGuard or an SSH tunnel, deliberately rather than by
  default.
- `ulimits.nofile: 262144`.

**What breaks without it.** Queryability, and only queryability. The plugin's `/api/audit/*`
routes answer 503 (they never degrade to an empty result — "nothing happened" and "we could not
look" are opposite answers to a security question). If an archive is configured, the record
itself is untouched: a sensor archives to S3 **directly**, not through the collector. Losing the
collector costs the ability to ask questions for a while; it never costs the record.

ClickHouse runs on the **collector** only, selected by the compose profile `collector`. A
`sensor` host runs Tetragon and Vector and ships its rows to the collector. Exactly one collector
per network.

---

## 2. How they are wired

```
  ┌──────────────────────────── one monitored host ────────────────────────────┐
  │                                                                            │
  │   kernel (eBPF)                                                            │
  │        │                                                                   │
  │        ▼                                                                   │
  │   ┌──────────┐   JSON lines     ┌─────────────────────────────┐            │
  │   │ tetragon │ ───────────────► │  $AUDIT_DATA_DIR/           │            │
  │   │          │  10MB x 5 rotate │    tetragon-log/*.log       │            │
  │   └──────────┘                  └─────────────┬───────────────┘            │
  │                                    tail (checkpointed, from beginning)     │
  │   journald (sshd/sudo/docker) ───────────┐    │                            │
  │                                          ▼    ▼                            │
  │                                  ┌───────────────────┐                     │
  │   astation plugin ──HTTP POST──► │      vector       │                     │
  │   (agent_tool / agent_llm /      │                   │                     │
  │    agent_session / audit_query)  │ normalise         │                     │
  │   http://audit-vector:8686/      │ redact secrets    │                     │
  │       ingest                     │ route by class    │                     │
  │   (audit docker network only)    └────────┬──────────┘                     │
  │                                           │                                │
  │                        ┌──────────────────┴───────────────────┐            │
  │                        │ 10 clickhouse sinks   │  aws_s3 sink │ OPTIONAL   │
  │                        ▼ (one per table)       ▼              │            │
  │                ┌───────────────┐       (compressed NDJSON, 60s batches)    │
  │                │  clickhouse   │              │                            │
  │                │  audit.*      │              │                            │
  │                │  (collector)  │              │                            │
  │                └───────┬───────┘              │                            │
  │      127.0.0.1:8124    │                      │                            │
  └────────────────────────┼──────────────────────┼────────────────────────────┘
                           │ reader (readonly)    │ put-only credential
                           │                      ▼
            astation plugin ──SELECT──┐    s3://<bucket>/raw/<class>/host=<label>/…
            /api/audit/* ─────────────┘    versioned, encrypted, Object Lock COMPLIANCE
            iOS app ──/api/audit/*──► astation        ▲
                                                      │ newest-object age
                        off-box heartbeat (a machine this host does not control)
```

Read it as four independent facts:

1. **Tetragon never talks to anything.** It writes a file. That is the whole interface, and it
   is why a broken shipper cannot break the sensor.
2. **Vector is the only component that writes.** Both sinks get *every* event. They are not a
   primary and a backup: ClickHouse is deletable by anything that gets root on the collector; S3
   is not deletable by anything at all.
3. **The plugin writes through Vector and reads around it.** It POSTs its own `agent_*` and
   `audit_query` rows to Vector's ingest endpoint, because the credential it holds for the store
   is read-only and cannot insert them. It reads with a direct ClickHouse connection as the
   `reader` user. Two different identities, two different directions.
4. **Liveness is judged from outside.** A quiet host produces no exec events, so "no data" and
   "dead sensor" are indistinguishable from the inside. `host_beat` is one synthetic event a
   minute, and an off-host watcher on a machine the audited host does not control
   watches the newest object under `raw/host_beat/host=<label>/`.

**Gateway-side settings** (the repo's own root `.env`, not this directory's):
`AUDIT_CLICKHOUSE_URL`, `AUDIT_CLICKHOUSE_USER=reader`, `AUDIT_CLICKHOUSE_PASSWORD`,
`AUDIT_INGEST_URL=http://audit-vector:8686/ingest`, `AUDIT_HOST_LABEL`, `AUDIT_QUERY_MAX_ROWS`,
`AUDIT_QUERY_TIMEOUT_S`. All are documented in the root `.env.example`, all are optional, and an
empty `AUDIT_CLICKHOUSE_URL` disables the audit routes entirely.

---

## 3. Configuration — every variable in `.env.example`

Copy `.env.example` to `.env` in this directory. Each variable is one of:

- **REQUIRED** — install fails without it.
- **DEFAULTED** — a blank or absent value gets a working default; a value you set is never overridden.
- **GENERATED** — `install.sh` invents one when blank and writes it back into `.env`, so re-running
  is idempotent and the value stays findable.
- **OPTIONAL** — blank is a supported, meaningful configuration.

| Variable | Status | What it does |
|---|---|---|
| `AUDIT_HOST_LABEL` | **REQUIRED** | This host's name in every row and every S3 key. Alphanumeric, dash, underscore. **Stable for the life of the host** — renaming splits its history in two and nothing joins the halves back. Deliberately not defaulted from the machine's hostname, because a hostname can change and this must not. |
| `AUDIT_ROLE` | DEFAULTED `collector` | `collector` = tetragon + vector + clickhouse. `sensor` = tetragon + vector. Selects the compose profile. |
| `AUDIT_DATA_DIR` | DEFAULTED `/opt/audit/data` | Export log, shipper buffers, and (collector) the store. Preflight refuses a collector with under 20G free, a sensor with under 5G. |
| `AUDIT_BUCKET` | **OPTIONAL** | The S3 archive bucket. **Empty = local-only mode**, and the `aws_s3` sink is left out of the rendered shipper config entirely. See [§8](#8-local-only-vs-archived). |
| `AWS_REGION` | DEFAULTED `us-east-1` | Only read when `AUDIT_BUCKET` is set. |
| `AUDIT_SHIPPER_ACCESS_KEY_ID` | OPTIONAL | The `audit-shipper` credential. `PutObject` under `raw/` and nothing else — no list, no read, no delete, no retention change. Assume anything that compromises the host can read this key; that scoping is why a compromised host still cannot erase what it already shipped. |
| `AUDIT_SHIPPER_SECRET_ACCESS_KEY` | OPTIONAL | Its secret. Blank with a bucket set means the shipper falls back to ambient host AWS credentials; `install.sh` warns. |
| `AUDIT_CLICKHOUSE_ENDPOINT` | DEFAULTED on a collector (`http://audit-clickhouse:8123`); **REQUIRED on a sensor** | Where the shipper sends rows. A sensor gets no default on purpose — a guess would silently point it at a store that is not there. |
| `AUDIT_CLICKHOUSE_HTTP_PORT` | DEFAULTED `8124` | The loopback-published port on the collector, for `install.sh` and the gateway. |
| `AUDIT_CLICKHOUSE_INGEST_PASSWORD` | GENERATED | The `ingest` user. Needed on every host, including sensors. |
| `AUDIT_CLICKHOUSE_READER_PASSWORD` | GENERATED (collector only) | The `reader` user. What the plugin and the AI analyst get. |
| `AUDIT_CLICKHOUSE_ADMIN_PASSWORD` | GENERATED (collector only) | The `admin` user. Schema and TTLs; loopback only. |

**Two values that look like settings and are not.** The shipper's disk buffer (1 GiB per sink)
and its ingest port (8686) are hard-coded in `vector/vector.yaml`. Vector parses its YAML before
interpolating and only substitutes into strings, so a typed integer or socket-address field cannot
read an environment variable at all. Change them in that file.

A note on multi-host installs: if you drive several machines from one workstation, keep the list of
them on the workstation and never on a monitored host. An inventory of every audited address has no
business sitting on a box that might be compromised.

---

## 4. Running it

Everything runs from this directory, on the host being audited.

```bash
cp .env.example .env          # then set AUDIT_HOST_LABEL; everything else is optional
./install.sh --preflight      # can this host run it? changes nothing, prints the mode
./install.sh                  # install or update, then self-test
./install.sh --no-test        # same, skipping the self-test (CI, or a slow box)
./selftest.sh                 # prove the pipeline works end to end, right now
./install.sh --uninstall      # stop and remove the containers; keep all the data
```

`install.sh` is idempotent — safe to re-run, and re-running after a partial failure completes the
remainder. It does, in order: validate config and apply defaults; check docker, root/sudo, BTF,
cgroup v2 and free disk; print what this host will run and whether an archive is configured;
generate any missing store passwords; create the data directories; render the shipper config;
`docker compose up -d --force-recreate`; verify every policy on disk is actually loaded; apply
`clickhouse/schema.sql` and the retention TTLs; verify the shipper is running; run `selftest.sh`.

`selftest.sh` is the stack's acceptance test — 21 checks, each of
which performs a **real** action and then goes looking for the record of it. A pass means the whole
chain worked (kernel probe → export file → shipper → transform → store), not that a container is
"up". It is non-destructive: it writes and deletes its own marker files and makes one outbound
connection to a public address. Run it when someone asks whether a host is really being audited.

**From the dev machine**, for a fleet, the same thing is one command per host — see `README.md`.
The installer validates every config file before applying anything,
which is deliberate: a malformed Tetragon policy takes the whole daemon down.

---

## 5. How the installer works

Two things in it are not obvious and are the reason it exists at all.

**It renders the shipper config; it does not template it at runtime.** `vector/vector.yaml` is a
template in which every per-host value is an `@@TOKEN@@` — `@@HOST_LABEL@@`, `@@BUCKET@@`,
`@@CLICKHOUSE_ENDPOINT@@`, `@@CLICKHOUSE_INGEST_PASSWORD@@`, `@@ROLE@@`, `@@AWS_REGION@@`,
`@@VECTOR_VERSION@@`. `install.sh` substitutes them into `vector/vector.rendered.yaml` (mode 600,
gitignored) and **fails if any token survives**. There is no `${VAR}` anywhere in that file on
purpose; [§9](#9-troubleshooting--the-traps-that-are-already-known) says why. Adding a new per-host
value means an `@@TOKEN@@` there and a `-e` line here.

**It forces container recreation.** `docker compose up -d --force-recreate`, every time. The audited hosts
deploy installs a bundle by renaming a directory into place, and a running container's bind mounts
stay attached to the **old** inode — so without recreation a container keeps serving the previous
config while the files on disk look correct. The cost is a few seconds of sensor downtime per
deploy, which is visible as a small hole in the archive. That is the honest trade.

---

## 6. The three ClickHouse identities

Defined in `clickhouse/users.d/audit.xml`. Passwords come from the environment, never from that
file — it is tracked in git and the host's `.env` is not.

| User | Can | Cannot | Lives where |
|---|---|---|---|
| `ingest` | `INSERT ON audit.*` | read a single row back, drop a table, see what is stored | the shipper's environment on **every** monitored host |
| `reader` | `SELECT ON audit.*`, capped at 10 000 rows / 30 s / 2 GB | write anything, DDL, raise its own caps | the astation gateway and the AI analyst |
| `admin` | everything, `WITH GRANT OPTION` | connect from anywhere but `127.0.0.1` / `::1` | only `install.sh`, on the collector itself |

The reasoning, in one line each:

- **`ingest` is treated as public.** It sits in a file on every monitored host, including hosts
  that may be compromised. Scoping it to insert-only means holding it buys an attacker the ability
  to add noise, not to read the record or erase it.
- **`reader`'s caps are `constraints`, not settings.** `readonly=1` alone still permits per-query
  settings changes in some modes, which is exactly how a caller raises its own ceiling and turns
  the limits into decoration. The `<constraints>` block makes `max_result_rows`,
  `max_execution_time`, `max_memory_usage` and `readonly` un-relaxable by any query.
- **`admin` is loopback-only.** The one credential that can drop data never accepts a connection
  from another machine.
- **The image's passwordless `default` superuser is removed** (`<default remove="remove"/>`) *and*
  prevented from being created (`CLICKHOUSE_SKIP_USER_SETUP=1`). Left enabled it is an
  unauthenticated writer on the network the sensors share.

`selftest.sh` check 11 proves all of this at runtime: `reader` attempting an INSERT must be
refused, `ingest` attempting a SELECT must be refused.

---

## 7. Retention

**`retention.yaml` is the only place retention is decided.** the retention renderer turns
it into the two systems that enforce it, which cannot see each other:

| Tier | Where | Enforced by |
|---|---|---|
| **hot** (`hot_days`) | ClickHouse on the collector's local disk, fast SQL | `ALTER TABLE … MODIFY TTL`, applied by `install.sh` |
| **warm** (`warm_days`) | the raw NDJSON archive in S3 Standard-IA, queryable by reload | an S3 lifecycle rule |
| **cold** (`cold_days`) | the archive in Glacier Deep Archive, then expiry | the same lifecycle rule |

Warm and cold are counted from the object's creation, not from the end of hot. `cold_days` is the
outer bound on the whole system's memory, and must be at least `object_lock_days` (30) — a
lifecycle rule cannot expire an object the Object Lock still protects, and the renderer refuses a
value below it. It also refuses a transition scheduled after the expiry and a zero-day TTL: silent
in production, loud here.

The windows are per event class, because the classes are nothing alike. `host_file` is the highest
volume by far (a build or a training run writes millions of files) and gets a short hot window with
the archive carrying the long tail. `host_exec` is the spine of every investigation and is cheap, so
it stays hot longer. `audit_query` — who asked the audit store what — is never short: the analyst
must stay audited for at least as long as the data it can read.

To change retention: edit `retention.yaml`, then

```bash
./install.sh          # re-renders retention and re-applies it
```

A table created by `schema.sql` has **no TTL** until the renderer applies one. `install.sh` does it
as its last store step; `selftest.sh` check 10 fails if any table is left without one.

The S3 key layout is `raw/<class>/host=<label>/%Y/%m/%d/%H/`. **Class comes first** so that one
lifecycle policy covers every host in the network without being edited when a host is added.

---

## 8. Local-only vs archived

`AUDIT_BUCKET` is the switch, and it is the only thing that decides.

| | `AUDIT_BUCKET` empty | `AUDIT_BUCKET` set |
|---|---|---|
| Containers | tetragon, vector, clickhouse | same |
| Shipper sinks | 10 ClickHouse sinks | 10 ClickHouse sinks + `aws_s3` |
| Capture, redaction, schema, TTLs, query surface | identical | identical |
| Needs a cloud account | no | yes, one bucket |
| Tamper-evident | **no** | yes |

In local-only mode `install.sh` deletes the `aws_s3` sink from the rendered config rather than
leaving it configured with an empty bucket, then asserts it is gone. An empty bucket name is
accepted by Vector and then retried against forever, which is the same silent-failure shape as the
`@@TOKEN@@` bug in [§9](#9-troubleshooting--the-traps-that-are-already-known).

**What you give up is the point of the whole design.** With an archive, a host that is fully
compromised can stop *new* records being written but cannot alter or delete what already landed —
and the gap it leaves is itself the alarm. Without one, the only copy of a record lives on the
machine that produced it, so root on that machine can erase its own history. Local-only is the
right mode for trying the stack out, for a lab box, or for a machine whose record does not need to
outlive it. It is not evidence.

One smaller consequence of local-only, stated because it is otherwise invisible: events whose class
matches no route — `route._unmatched`, which is where the `agent_unknown` bucket lands — have no
sink at all and are dropped. With an archive they still reach S3, because the `aws_s3` sink is the
only consumer of `route._unmatched`.

Turning the archive on later is not a migration: create the bucket once for the
network, put the bucket and the shipper key in `.env`, re-run `./install.sh`. Events from before
the switch stay in ClickHouse under their hot TTL and are simply not in the archive.

---

## 9. Troubleshooting — the traps that are already known

These are not hypotheticals. Every one was found by the validation rather than by review, and
every one was **silent or misleadingly reported** — "the container is up" was true during most of
them.

**Vector does not interpolate environment variables.** Measured on 0.58.0, not assumed. Typed
fields fail loudly ("invalid socket address syntax" for a port, "invalid uri character" for an
endpoint, "did not match any variant of untagged enum BufferConfig" for an integer). String fields
fail **silently**, which is far worse: 738 rows were stored with `host` set to the literal
`${AUDIT_HOST_LABEL}` while the S3 sink retried against a bucket actually named
`${AUDIT_BUCKET}`. This is why every per-host value in `vector/vector.yaml` is an `@@TOKEN@@`
rendered by `install.sh`, and why the installer fails if one survives. **Do not add a `${VAR}` to
that file.**

**A malformed Tetragon policy takes the whole daemon down**, not just that policy — the host is
left with no sensor at all. Observed with five values in one selector: Tetragon refuses more than
four and then refuses to boot. `install.sh` therefore checks the daemon is alive *and* that every
policy file on disk is actually loaded, and the installer parses every config before
shipping anything.

**A `--` inside an XML comment breaks ClickHouse's users file.** A double hyphen is illegal inside
an XML comment, and ClickHouse refuses the whole file rather than the comment. Rephrase, or use a
single hyphen; never put a double hyphen in a comment in `clickhouse/users.d/audit.xml` or
`clickhouse/config.d/audit.xml`.

**`<grants>` beside `<access_management>` on one user is a restart loop.** ClickHouse refuses the
combination ("Any other access control settings can't be specified with `grants`") and the reason
appears only in `clickhouse-server.err.log`. That is why the `admin` user has `GRANT ALL ON *.*
WITH GRANT OPTION` and no `<access_management>` element. It is also why the log directory is
mounted out of the container: a container in a restart loop takes its own logs with it.

**Bind mounts go stale after a directory `mv`.** A running container's mounts are bound to the old
directory's *inode*, so after the deploy renames a new bundle into place the container keeps
serving the previous config while the files on disk look correct. ClickHouse hit this and silently
had no `users.d` at all — it rejected every password, including an empty one, while the correct
file sat right there on the host. `install.sh` passes `--force-recreate` for exactly this reason.
**If you ever `mv` this directory, recreate the containers.**

**Store passwords that never reach the container.** `users.d/audit.xml` reads all three with
`from_env`, so they must be in the ClickHouse container's environment, not merely in `.env`.
Without them the users exist with an empty password and every login fails, including the
installer's own.

**`current_boot_only: false` is rejected on systemd 250–257** and the refusal is fatal to the whole
shipper, not just the journald source. Leave it `true`.

**VRL limits that look like bugs.** It has no user-defined functions (`redact = func(s) {…}`
compiles in no version and takes the shipper down, which is why the redaction chain is repeated per
field rather than factored out) and no `to_timestamp` (timestamps are type-checked with
`is_timestamp`, not converted). Separately, `system.tables` has no `ttl_expression` column — the
TTL only shows up inside `create_table_query`, which is what `selftest.sh` matches on.

**`decoding.codec`, not `encoding:`, on the HTTP source.** With the older key the source did not
parse the body at all: every field arrived as one string in `.message`, the class silently
defaulted, and 44 rows landed with every column empty. The transform parses defensively as well, so
neither half alone can bring that back.

**Where to look when something is wrong.**

```bash
docker compose --profile collector ps
docker logs --since 10m audit-tetragon
docker logs --since 10m audit-vector          # VECTOR_LOG=warn; `info` narrates every batch
docker exec audit-tetragon tetra tracingpolicy list
grep -hE '<Error>' $AUDIT_DATA_DIR/clickhouse-log/clickhouse-server.err.log | tail
./selftest.sh                                  # the honest answer, end to end
```

Two symptoms with non-obvious causes: **shipper lag is a correctness measure, not a performance
one** — Tetragon keeps only 5 × 10 MB of export, so a shipper more than ~50 MB behind is losing
events before they are ever read, which is why `selftest.sh` check 8 measures it. And **parse
failures are kept, not dropped** — they arrive as `host_beat` rows whose `note` starts with
`PARSE FAILURE`, so a sensor whose output shape changed on upgrade shows up as a rising count
rather than as events that quietly stop.

---

## 10. Files in this directory

| Path | What it is |
|---|---|
| `AUDIT_SETUP.md` | this file — the full explanation |
| `README.md` | the short operator card: add a host, run it, change retention |
| `docker-compose.yml` | the three services. Images pinned by digest. `AUDIT_ROLE` selects the profile. |
| `.env.example` | every setting, marked required / defaulted / generated / optional |
| `install.sh` | idempotent installer: preflight, secrets, render, up, verify, self-test |
| `selftest.sh` | 21 end-to-end checks, each performing a real action and finding its record |
| `retention.yaml` | the only place retention is decided |
| `tetragon/policies/` | what the kernel watches. Read the header of `file-mutations.yaml` before changing any of it; the selectors are narrower than they look, for reasons the comments explain. |
| `vector/vector.yaml` | the shipper **template**; `install.sh` renders it to `vector.rendered.yaml` |
| `clickhouse/schema.sql` | one table per event class, `CREATE … IF NOT EXISTS` |
| `clickhouse/users.d/` | the three identities |
| `clickhouse/config.d/` | store-level limits: log size, query-log TTL, memory ratio |
