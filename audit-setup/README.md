# audit-setup — host telemetry for every machine on this network

Records what happens on a host at the kernel: every process executed, every file created,
written, renamed or deleted, and every TCP connection made or accepted. Ships it to a queryable
store and, optionally, to an append-only archive the host itself cannot alter.

**This file is the operator card — what to type.** For what the three containers are, why they
were chosen, how they are wired, every setting, and every trap already found the hard way, read
**[`AUDIT_SETUP.md`](AUDIT_SETUP.md)**. Design and rationale: `../docs/AUDIT_DESIGN.md`. Build
plan and measurements: `../docs/AUDIT_PLAN.md`.

## Run it on one host

```bash
cp .env.example .env          # set AUDIT_HOST_LABEL; every other value is optional
./install.sh --preflight      # can this host run it? changes nothing
./install.sh                  # install/update, then self-test
```

That is a complete working stack — sensor, shipper, queryable store — with no AWS account. It
runs **local-only**: nothing leaves the machine, so the record is not tamper-evident. Set
`AUDIT_BUCKET` to add the archive ([`AUDIT_SETUP.md` §8](AUDIT_SETUP.md#8-local-only-vs-archived)).

## Add a host to the fleet

Three steps, and the third is one command.

1. **Once per network** — create the bucket and the two credentials:

   ```bash
   scripts/audit_bootstrap_aws.sh          # idempotent; safe to re-run
   scripts/audit_verify_aws.sh             # 17 adversarial checks, all must pass
   ```

2. **Add the host** to `audit-setup/hosts.yaml` (gitignored; copy
   `hosts.example.yaml`). One entry:

   ```yaml
   - label: gpu4090            # stable forever: it is in every row and every S3 key
     ssh: user@host
     ssh_key: ~/.ssh/some_key  # optional
     role: sensor              # or `collector` for the one host with the store
   ```

3. **Install:**

   ```bash
   scripts/deploy_audit.sh gpu4090
   ```

   That validates the bundle locally, ships it, installs it, and runs the
   self-test. There is no fourth step and no runbook — if a host ever needs one,
   fix `install.sh` instead of writing it down.

Verify from the dev machine, which is the only judgement a compromised host cannot influence:

```bash
uv run --project services/research-gateway python scripts/audit_heartbeat.py --print-age
```

## The two roles

| | `collector` | `sensor` |
|---|---|---|
| Runs | tetragon + vector + clickhouse | tetragon + vector |
| Archive to S3 | yes, its own | yes, its own |
| Rows go to | its own store | the collector's store |
| How many | exactly one per network | the rest |

A sensor archives to S3 **directly**, not through the collector. Losing the
collector costs queryability for a while; it never costs the record.

## Is a host really being audited?

```bash
./selftest.sh      # 21 checks, each performs a real action and then finds its record
```

A pass means the whole chain worked — kernel probe, export file, shipper, transform, store — not
that a container is "up". Non-destructive.

## Change retention

Edit `retention.yaml` — the only place retention is decided — then:

```bash
uv run --project services/research-gateway python scripts/audit_retention.py check
scripts/deploy_audit.sh <label>      # re-renders and re-applies
```

The renderer refuses windows that S3 or ClickHouse would accept and then fail to
enforce — an expiry inside the Object Lock window, a transition scheduled after
the expiry, a zero-day TTL. Those are silent in production and loud here.

## Attack it

```bash
scripts/audit_attack.sh <label>
```

Stops the sensor and the shipper from *inside* the audited container, then
proves the archive is still intact and still undeletable. It pauses auditing
while it runs and restores everything at the end.

## Four things that will bite you

The full list, with the diagnosis for each, is in
[`AUDIT_SETUP.md` §9](AUDIT_SETUP.md#9-troubleshooting--the-traps-that-are-already-known). The
four worth knowing before you touch anything here:

1. **Do not add a `${VAR}` to `vector/vector.yaml`.** Vector does not interpolate environment
   variables in this version, and string fields fail *silently*. Per-host values are `@@TOKEN@@`
   markers rendered by `install.sh`.
2. **A malformed Tetragon policy takes the whole daemon down**, leaving the host with no sensor
   at all. `install.sh` checks every policy on disk is loaded; `deploy_audit.sh` parses every
   config before shipping.
3. **Containers must be recreated on deploy.** A running container's bind mounts stay attached to
   the old inode after a directory `mv`, so it keeps serving the previous config while the files
   on disk look right. Hence `--force-recreate`.
4. **Retention has to be rendered before it exists.** A table from `schema.sql` has no TTL until
   `scripts/audit_retention.py` applies one. `install.sh` does it; `selftest.sh` fails if any
   table is left without one.

## Uninstall

```bash
./install.sh --uninstall      # stops and removes the containers; keeps all the data
```
