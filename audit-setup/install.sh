#!/usr/bin/env bash
# Install or update the audit stack ON THIS HOST. Idempotent: safe to re-run,
# and re-running after a partial failure completes the remainder.
#
#   ./install.sh                 # install/update, then self-test
#   ./install.sh --preflight     # check this host can run it; change nothing
#   ./install.sh --no-test       # skip the self-test (CI, or a slow box)
#   ./install.sh --uninstall     # stop and remove containers; keep the data
#
# It is run BY scripts/deploy_audit.sh from the dev Mac, which copies this
# directory to the host first. Running it here by hand is equally supported --
# that is the point of keeping every host-specific value in .env.
#
# Adding host number four is: fill in .env, run this, done. If that stops being
# true, fix it here rather than writing a runbook step.
set -euo pipefail
cd "$(dirname "$0")"

ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
made() { printf '  \033[36mdo\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
die()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }

MODE=install
case "${1:-}" in
  --preflight) MODE=preflight ;;
  --no-test)   MODE=notest ;;
  --uninstall) MODE=uninstall ;;
  "") ;;
  *) die "unknown option: $1" ;;
esac

# --- 0. configuration ------------------------------------------------------
# ONE value is genuinely required: AUDIT_HOST_LABEL. Everything else either has
# a default that works, or is generated here. AUDIT_BUCKET is OPTIONAL: leave it
# empty and the host runs local-only -- sensor, shipper and hot store, no S3 --
# which is a complete working stack for someone who has no AWS account and has
# not run scripts/audit_bootstrap_aws.sh.
[ -f .env ] || die ".env is missing. Copy .env.example and fill it in."
set -a; . ./.env; set +a

: "${AUDIT_HOST_LABEL:?set AUDIT_HOST_LABEL in .env}"

# Defaults, applied only when the value is absent or empty. A value that IS set
# is never overridden, so a configured host behaves exactly as it did before.
AUDIT_ROLE="${AUDIT_ROLE:-collector}"
AUDIT_DATA_DIR="${AUDIT_DATA_DIR:-/opt/audit/data}"
AUDIT_BUCKET="${AUDIT_BUCKET:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AUDIT_CLICKHOUSE_HTTP_PORT="${AUDIT_CLICKHOUSE_HTTP_PORT:-8124}"
export AUDIT_ROLE AUDIT_DATA_DIR AUDIT_BUCKET AWS_REGION AUDIT_CLICKHOUSE_HTTP_PORT

case "$AUDIT_ROLE" in collector|sensor) ;; *) die "AUDIT_ROLE must be collector or sensor" ;; esac

# A collector holds the store, so it knows where the store is: its own container
# on the audit network. A sensor does not, and guessing would silently point it
# at a ClickHouse that is not there.
if [ -z "${AUDIT_CLICKHOUSE_ENDPOINT:-}" ]; then
  [ "$AUDIT_ROLE" = "collector" ] \
    || die "set AUDIT_CLICKHOUSE_ENDPOINT in .env -- a sensor must be told where the collector is"
  AUDIT_CLICKHOUSE_ENDPOINT="http://audit-clickhouse:8123"
fi
export AUDIT_CLICKHOUSE_ENDPOINT

# The label becomes a table value and an S3 key segment on every event this
# host ever ships. A character that needs escaping in one of those places is a
# problem discovered months later, in a query that returns nothing.
case "$AUDIT_HOST_LABEL" in
  *[!a-zA-Z0-9_-]*) die "AUDIT_HOST_LABEL must be alphanumeric, dash or underscore: '$AUDIT_HOST_LABEL'" ;;
esac

# Archive on or off, decided once and reported everywhere below.
ARCHIVE=off
if [ -n "$AUDIT_BUCKET" ]; then ARCHIVE=on; fi

# --- 1. preflight ----------------------------------------------------------
step "1. Can this host run the sensor?"

command -v docker >/dev/null || die "docker is not installed"
docker compose version >/dev/null 2>&1 || die "docker compose v2 is not available"
ok "docker $(docker --version | awk '{print $3}' | tr -d ,), compose $(docker compose version --short)"

[ "$(id -u)" = "0" ] || sudo -n true 2>/dev/null || die "needs root or passwordless sudo (privileged container, host pid namespace)"
ok "root or sudo available"

# eBPF without kernel headers needs BTF. Its absence is the one hard stop:
# every probe in tetragon/policies/ attaches by kernel function name.
[ -r /sys/kernel/btf/vmlinux ] || die "/sys/kernel/btf/vmlinux is missing -- this kernel has no BTF, so the eBPF probes cannot load"
ok "BTF present ($(stat -c %s /sys/kernel/btf/vmlinux 2>/dev/null || echo ?) bytes), kernel $(uname -r)"

[ "$(stat -fc %T /sys/fs/cgroup 2>/dev/null)" = "cgroup2fs" ] \
  || warn "cgroup v1 -- container attribution may be incomplete (v2 expected)"

# The store is a guest on a disk shared with the work it audits, and a full
# disk takes the audited workload down with it. Measured baseline on the first
# host: ~500k events/day idle, well under 1 GB/day in ClickHouse.
avail_gb=$(df -BG --output=avail "$(dirname "$AUDIT_DATA_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$avail_gb" ]; then
  if [ "$AUDIT_ROLE" = "collector" ] && [ "$avail_gb" -lt 20 ]; then
    die "only ${avail_gb}G free; a collector needs 20G+ for the hot store"
  elif [ "$avail_gb" -lt 5 ]; then
    die "only ${avail_gb}G free; a sensor needs 5G+ for the buffer and export log"
  fi
  ok "${avail_gb}G free on $(dirname "$AUDIT_DATA_DIR")"
fi

# --- what this host is about to run ----------------------------------------
# Printed in every mode, including --preflight, because "is my data leaving this
# machine or not" is the one question nobody should have to read a config to
# answer.
step "What this host will run"
echo "  host label   $AUDIT_HOST_LABEL"
echo "  role         $AUDIT_ROLE  ($([ "$AUDIT_ROLE" = collector ] && echo 'tetragon + vector + clickhouse' || echo 'tetragon + vector'))"
echo "  data dir     $AUDIT_DATA_DIR"
echo "  store        $AUDIT_CLICKHOUSE_ENDPOINT"
if [ "$ARCHIVE" = on ]; then
  ok "archive: s3://$AUDIT_BUCKET/raw/<class>/host=$AUDIT_HOST_LABEL/  (region $AWS_REGION)"
  [ -n "${AUDIT_SHIPPER_ACCESS_KEY_ID:-}" ] \
    || warn "AUDIT_BUCKET is set but AUDIT_SHIPPER_ACCESS_KEY_ID is empty -- the shipper will fall back to the host's ambient AWS credentials, if it has any"
else
  warn "LOCAL-ONLY MODE (NO ARCHIVE). AUDIT_BUCKET is empty, so the S3 sink is left"
  warn "  out of the shipper config entirely. Everything is still captured,"
  warn "  redacted and queryable in ClickHouse -- but the only copy of a record"
  warn "  lives on the machine that produced it, so root on this host can erase"
  warn "  its own history. Fine for trying the stack out or for a lab box."
  warn "  For tamper-evidence: run scripts/audit_bootstrap_aws.sh, then set"
  warn "  AUDIT_BUCKET and the shipper key in .env and re-run this installer."
fi

if [ "$MODE" = preflight ]; then
  step "Preflight only -- nothing changed."
  exit 0
fi

# --- 0b. secrets -----------------------------------------------------------
# A blank store password used to mean "every login fails", which is a confusing
# way to learn you were supposed to invent three passwords. Blank now means
# "generate one", written back to .env so re-running is idempotent and so the
# value is findable afterwards. A password that IS set is never touched.
gen_secret() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 24
  else
    head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n'
  fi
}

ensure_secret() {  # VAR_NAME
  local var="$1" val
  val="${!var:-}"
  [ -n "$val" ] && return 0
  val="$(gen_secret)"
  if grep -qE "^${var}=" .env; then
    sed -i.bak "s|^${var}=.*|${var}=${val}|" .env && rm -f .env.bak
  else
    printf '%s=%s\n' "$var" "$val" >> .env
  fi
  chmod 600 .env
  export "$var=$val"
  made "generated $var (written to .env)"
}

# --- uninstall -------------------------------------------------------------
if [ "$MODE" = uninstall ]; then
  step "Uninstalling (data in $AUDIT_DATA_DIR is KEPT)"
  docker compose --profile collector --profile sensor down 2>/dev/null || true
  ok "containers removed; $AUDIT_DATA_DIR untouched"
  if [ "$ARCHIVE" = on ]; then
    warn "the archive in S3 is unaffected and cannot be removed from this host by design"
  else
    warn "local-only mode: there is no archive, so $AUDIT_DATA_DIR is the only copy"
  fi
  exit 0
fi

# --- 1b. store passwords ---------------------------------------------------
step "1b. Store credentials"
# Only the collector runs ClickHouse, but a sensor's shipper still authenticates
# to the collector's store, so it needs the ingest password too -- and only that
# one. Generating a reader/admin password on a sensor would invent a credential
# that matches nothing.
ensure_secret AUDIT_CLICKHOUSE_INGEST_PASSWORD
if [ "$AUDIT_ROLE" = "collector" ]; then
  ensure_secret AUDIT_CLICKHOUSE_READER_PASSWORD
  ensure_secret AUDIT_CLICKHOUSE_ADMIN_PASSWORD
fi
ok "ingest/reader/admin passwords present"

# --- 2. directories --------------------------------------------------------
step "2. Data directories"
mkdir -p "$AUDIT_DATA_DIR"/{tetragon-log,vector,clickhouse,clickhouse-log}
# ClickHouse runs as uid 101 in its image and will not start on a root-owned
# volume it cannot write. The log directory needs it too, and a container that
# cannot write its log fails with no way to find out why.
if [ "$AUDIT_ROLE" = "collector" ]; then
  chown -R 101:101 "$AUDIT_DATA_DIR/clickhouse" "$AUDIT_DATA_DIR/clickhouse-log" 2>/dev/null || true
fi
ok "$AUDIT_DATA_DIR"

# --- 3. render the shipper config ------------------------------------------
step "3. Shipper config"
# Vector cannot interpolate an environment variable into a TYPED field (URI,
# socket address, integer): it parses the YAML before interpolating and only
# substitutes into strings. Those fields carry @@TOKEN@@ markers instead and
# are rendered here, before the file is mounted. See the header of
# vector/vector.yaml.
#
# The S3 sink is OPTIONAL and is removed here rather than left configured with
# an empty bucket -- Vector would accept that and then retry forever against a
# bucket named "", which is the same silent-failure shape as the @@TOKEN@@ bug
# above. The template brackets it with two comment markers; with a bucket set
# only the markers are dropped, so the rendered file is unchanged from before.
if [ "$ARCHIVE" = on ]; then
  ARCHIVE_SED='/@@ARCHIVE_BEGIN@@/d;/@@ARCHIVE_END@@/d'
else
  ARCHIVE_SED='/@@ARCHIVE_BEGIN@@/,/@@ARCHIVE_END@@/d'
fi
sed -e "$ARCHIVE_SED" vector/vector.yaml \
  | sed -e "s|@@CLICKHOUSE_ENDPOINT@@|${AUDIT_CLICKHOUSE_ENDPOINT}|g" \
    -e "s|@@CLICKHOUSE_INGEST_PASSWORD@@|${AUDIT_CLICKHOUSE_INGEST_PASSWORD}|g" \
    -e "s|@@HOST_LABEL@@|${AUDIT_HOST_LABEL}|g" \
    -e "s|@@ROLE@@|${AUDIT_ROLE}|g" \
    -e "s|@@BUCKET@@|${AUDIT_BUCKET}|g" \
    -e "s|@@AWS_REGION@@|${AWS_REGION:-us-east-1}|g" \
    -e "s|@@VECTOR_VERSION@@|${VECTOR_VERSION:-0.58.0}|g" \
    > vector/vector.rendered.yaml
# The rendered file holds the ingest password in clear text, like the .env it
# came from. Mounted read-only into the container; not readable by anyone else.
chmod 600 vector/vector.rendered.yaml
# Comment lines are excluded: the file's own header explains the @@TOKEN@@
# convention and would otherwise trip this check every time.
if grep -vE '^\s*#' vector/vector.rendered.yaml | grep -q '@@'; then
  grep -nvE '^\s*#' vector/vector.rendered.yaml | grep '@@' | head -3
  die "unrendered @@TOKEN@@ left in the shipper config"
fi
# The archive block leaving means the aws_s3 sink leaves with it. Asserted
# rather than assumed: a marker typo would otherwise ship events to a bucket
# named "" and only show up as retries in the shipper's log.
if [ "$ARCHIVE" = on ]; then
  grep -q 'type: aws_s3' vector/vector.rendered.yaml || die "archive requested but the aws_s3 sink is not in the rendered config"
else
  ! grep -q 'type: aws_s3' vector/vector.rendered.yaml || die "local-only mode but the aws_s3 sink survived rendering"
fi
ok "rendered ($(grep -c 'endpoint:' vector/vector.rendered.yaml) endpoints -> $AUDIT_CLICKHOUSE_ENDPOINT, archive $ARCHIVE)"

# --- 4. bring the stack up -------------------------------------------------
step "4. Containers (role: $AUDIT_ROLE)"
COMPOSE_PROFILE=""
if [ "$AUDIT_ROLE" = "collector" ]; then COMPOSE_PROFILE="--profile collector"; fi

# Pinned by digest, so this pulls exactly what the compose file names and
# nothing moves under us.
docker compose $COMPOSE_PROFILE pull --quiet 2>/dev/null || warn "pull failed; using local images"

# --force-recreate is REQUIRED, not a precaution. deploy_audit.sh installs a new
# bundle by renaming the directory into place, and a running container's bind
# mounts are bound to the OLD directory's inode -- so without recreation a
# container keeps serving the previous bundle while the files on disk look
# correct. That failure is silent and confusing: ClickHouse could not see its
# own users.d, so it had NO admin user at all and rejected every password,
# including an empty one, while the file sat right there on the host.
#
# The cost is a few seconds of sensor downtime per deploy. That gap is real and
# will be visible as a hole in the archive, which is the honest trade.
docker compose $COMPOSE_PROFILE up -d --force-recreate
made "docker compose up (containers recreated so bind mounts re-resolve)"

# --- 4. the sensor actually loaded its policies ----------------------------
step "5. Sensor"
# A malformed policy does not disable itself -- it takes the WHOLE DAEMON down
# at startup (observed: five values in one selector, Tetragon refused to boot).
# So this checks the daemon is alive AND that every policy file on disk is
# loaded, rather than trusting `up -d` to have meant something.
for i in $(seq 1 30); do
  state=$(docker inspect -f '{{.State.Status}}' audit-tetragon 2>/dev/null || echo missing)
  [ "$state" = "running" ] && break
  sleep 2
done
[ "$state" = "running" ] || die "audit-tetragon is '$state'. docker logs audit-tetragon"

sleep 8
loaded=$(docker exec audit-tetragon tetra tracingpolicy list 2>/dev/null | awk 'NR>1 && $3=="enabled" {print $2}' | sort | tr '\n' ' ')
expected=$(ls tetragon/policies/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml$//' | sort | tr '\n' ' ')
if [ -z "$loaded" ]; then
  docker logs audit-tetragon 2>&1 | grep -i "failed to execute\|error" | tail -3
  die "no tracing policies loaded -- the daemon usually refuses to start on a malformed one"
fi
ok "policies enabled: $loaded"
[ "$loaded" = "$expected" ] || warn "expected [$expected] but have [$loaded]"

# --- 5. schema + retention (collector only) --------------------------------
if [ "$AUDIT_ROLE" = "collector" ]; then
  step "6. Store"
  for i in $(seq 1 45); do
    docker exec audit-clickhouse wget --spider -q http://127.0.0.1:8123/ping 2>/dev/null && break
    sleep 2
  done
  docker exec audit-clickhouse wget --spider -q http://127.0.0.1:8123/ping 2>/dev/null \
    || { docker logs audit-clickhouse 2>&1 | tail -5
         echo "  --- clickhouse-server.err.log ---"
         # Drop the numbered stack frames; the diagnosis is the <Error> line.
         grep -hE '<Error>' "$AUDIT_DATA_DIR/clickhouse-log/clickhouse-server.err.log" 2>/dev/null \
           | grep -vE '^[0-9]+\. ' | tail -3 | cut -c1-300
         die "clickhouse did not come up"; }
  ok "clickhouse responding"

  # Every statement is CREATE IF NOT EXISTS, so this runs on every install.
  docker exec -i audit-clickhouse clickhouse-client \
    --user admin --password "$AUDIT_CLICKHOUSE_ADMIN_PASSWORD" \
    --multiquery < clickhouse/schema.sql
  tables=$(docker exec audit-clickhouse clickhouse-client --user admin \
    --password "$AUDIT_CLICKHOUSE_ADMIN_PASSWORD" \
    --query "SELECT count() FROM system.tables WHERE database='audit'")
  ok "schema applied ($tables tables)"

  # Retention is decided in audit-setup/retention.yaml and applied here, so a fresh
  # table is never left with no TTL. The renderer refuses windows that S3 or
  # ClickHouse would silently fail to enforce.
  if [ -f retention.ttl.sql ]; then
    docker exec -i audit-clickhouse clickhouse-client --user admin \
      --password "$AUDIT_CLICKHOUSE_ADMIN_PASSWORD" --multiquery < retention.ttl.sql
    ok "retention TTLs applied ($(grep -c 'MODIFY TTL' retention.ttl.sql) tables)"
  else
    warn "retention.ttl.sql not shipped -- tables have NO TTL and will grow without bound."
    warn "  render it on the Mac: uv run --project services/research-gateway \\"
    warn "    python scripts/audit_retention.py ttl > audit-setup/retention.ttl.sql"
  fi
fi

# --- 6. shipper ------------------------------------------------------------
step "7. Shipper"
for i in $(seq 1 20); do
  state=$(docker inspect -f '{{.State.Status}}' audit-vector 2>/dev/null || echo missing)
  [ "$state" = "running" ] && break
  sleep 2
done
[ "$state" = "running" ] || { docker logs audit-vector 2>&1 | tail -10; die "audit-vector is '$state'"; }
ok "vector running"

step "Installed"
echo "  host label   $AUDIT_HOST_LABEL"
echo "  role         $AUDIT_ROLE"
if [ "$ARCHIVE" = on ]; then
  echo "  archive      s3://$AUDIT_BUCKET/raw/<class>/host=$AUDIT_HOST_LABEL/"
else
  echo "  archive      none (local-only mode; set AUDIT_BUCKET in .env to enable)"
fi
if [ "$AUDIT_ROLE" = "collector" ]; then
  echo "  store        http://127.0.0.1:${AUDIT_CLICKHOUSE_HTTP_PORT}  (user: reader)"
fi

if [ "$MODE" != notest ]; then
  step "Self-test"
  ./selftest.sh
fi
