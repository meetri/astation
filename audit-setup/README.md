# audit-setup

The host half of the audit layer: a kernel sensor, a shipper and a queryable store, as a Compose
project.

It records what happens on a machine at the kernel level — every process executed, every file
created, written, renamed or deleted, and every TCP connection made or accepted — and makes it
queryable. Optionally it also writes to an append-only archive the host itself cannot alter.

The plugin's own half, which ties that activity to the agent session that caused it, is in the
parent directory and needs nothing from here to run.

**This file is the quick start.** For what each container is, why it was chosen, how they are
wired, and what every setting does, read [`AUDIT_SETUP.md`](AUDIT_SETUP.md).

## Quick start

```bash
cp .env.example .env          # set AUDIT_HOST_LABEL; everything else has a default
./install.sh --preflight      # check this host can run it; changes nothing
./install.sh                  # install, then self-test
```

That is a complete working stack with no cloud account: sensor, shipper and queryable store. It
runs **local-only**, which means nothing leaves the machine and the record is therefore not
tamper-evident. Set `AUDIT_BUCKET` to add the archive — see
[§8](AUDIT_SETUP.md#8-local-only-vs-archived).

## Connecting the plugin

The plugin reads the store directly and writes session events through the shipper. Point it at
both:

```bash
AUDIT_INGEST_URL=http://<shipper-host>:8686/ingest
AUDIT_CLICKHOUSE_URL=http://<store-host>:8123
AUDIT_CLICKHOUSE_USER=reader
AUDIT_CLICKHOUSE_PASSWORD=<the reader password from .env>
AUDIT_HOST_LABEL=<this host's label>
```

Both services must be reachable from wherever the agent runs. With neither set, the plugin starts
normally and its audit surface reports that it is not configured.

## Verifying

```bash
./selftest.sh
```

21 checks against the running stack: that an in-container process carries a container id, that file
creates, renames and deletes are all captured, that paths resolve, that inbound and outbound
connections are recorded, that a secret is masked rather than dropped, that the reader role cannot
write, that the ingest role cannot read, and that every table has a retention policy.

## Updating the configuration

`install.sh` is idempotent. Edit `.env` or any config file and run it again; it re-renders,
recreates what changed and re-runs the self-test.

```bash
./install.sh --no-test        # skip the self-test
./install.sh --uninstall      # stop and remove containers, keep the data
```

## Retention

`retention.yaml` is the only place retention is decided. It sets how long each class of event stays
in the queryable store, and how long archived copies are kept before moving to colder storage or
expiring. Edit it and re-run `install.sh`.

## Layout

| Path | What it is |
|---|---|
| `docker-compose.yml` | The three services and their profiles |
| `install.sh` | Renders config, starts the stack, runs the self-test |
| `selftest.sh` | 21 checks against a running stack |
| `.env.example` | Every setting, with which are required |
| `retention.yaml` | Retention policy, per event class |
| `tetragon/policies/` | What the kernel sensor watches |
| `vector/vector.yaml` | Parsing, redaction, routing. A template |
| `clickhouse/` | Schema, roles and server settings |
