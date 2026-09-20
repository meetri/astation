#!/usr/bin/env bash
# Prove this host's audit pipeline actually works, end to end, right now.
#
#   ./selftest.sh
#
# This is the Phase 1 validation table of docs/AUDIT_PLAN.md as a command. Each
# check performs a REAL action and then goes looking for the record of it, so a
# pass means the whole chain worked -- kernel probe, export file, shipper,
# transform, store -- not that a container is "up".
#
# It runs after every install, and is the thing to run when someone asks
# whether a host is really being audited. Nothing here is destructive: it
# writes and deletes its own marker files under /tmp and makes one outbound
# connection to a public address.
set -uo pipefail
cd "$(dirname "$0")"
set -a; . ./.env; set +a
# The same defaults install.sh applies, so a minimal .env (label only) is as
# testable as a fully written one. A value that IS set is never overridden.
AUDIT_ROLE="${AUDIT_ROLE:-collector}"
AUDIT_BUCKET="${AUDIT_BUCKET:-}"

pass=0; fail=0; skip=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$*"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=$((fail+1)); }
note() { printf '  \033[33mskip\033[0m  %s\n' "$*"; skip=$((skip+1)); }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }

MARK="audit-selftest-$$-$(date +%s)"
CH() { docker exec audit-clickhouse clickhouse-client --user admin \
        --password "$AUDIT_CLICKHOUSE_ADMIN_PASSWORD" --query "$1" 2>/dev/null; }

# The store is eventually consistent by design (Vector batches for 5s), so a
# check that looks once and fails is measuring its own impatience. Every query
# below is retried to a deadline instead.
await() {  # seconds query expected_min description
  local deadline=$(( $(date +%s) + $1 )) q="$2" want="$3" desc="$4" got=0
  while [ "$(date +%s)" -lt "$deadline" ]; do
    got=$(CH "$q" | tr -dc '0-9'); got=${got:-0}
    [ "$got" -ge "$want" ] && { ok "$desc (found $got)"; return 0; }
    sleep 2
  done
  bad "$desc -- wanted >=$want, found $got after $1s"
  return 1
}

step "Containers"
for c in audit-tetragon audit-vector; do
  s=$(docker inspect -f '{{.State.Status}}' "$c" 2>/dev/null || echo missing)
  [ "$s" = running ] && ok "$c running" || bad "$c is $s"
done
if [ "$AUDIT_ROLE" = collector ]; then
  s=$(docker inspect -f '{{.State.Status}}' audit-clickhouse 2>/dev/null || echo missing)
  [ "$s" = running ] && ok "audit-clickhouse running" || bad "audit-clickhouse is $s"
fi

step "Sensor policies"
loaded=$(docker exec audit-tetragon tetra tracingpolicy list 2>/dev/null | awk 'NR>1 && $3=="enabled"' | wc -l)
[ "${loaded:-0}" -ge 2 ] && ok "$loaded tracing policies enabled" || bad "only ${loaded:-0} policies enabled"

if [ "$AUDIT_ROLE" != collector ]; then
  step "Store checks"
  note "sensor role: rows go to the collector, not checked from here"
  step "Archive"
  # A sensor cannot read its own archive back -- the credential is put-only by
  # design -- so liveness for these hosts is judged off-box by
  # scripts/audit_heartbeat.py on the Mac.
  note "put-only credential: archive is verified off-box by audit_heartbeat.py"
  printf '\n\033[1m%d passed, %d failed, %d skipped\033[0m\n' "$pass" "$fail" "$skip"
  [ "$fail" -eq 0 ] || exit 1
  exit 0
fi

# --- the real end-to-end checks -------------------------------------------

step "1. Process execution is recorded"
# A distinctive argv, so the row found is unambiguously the one this test made.
/bin/echo "$MARK-exec" >/dev/null
await 45 "SELECT count() FROM audit.host_exec WHERE arguments LIKE '%$MARK-exec%'" 1 \
  "exec captured with full argv"

step "2. Container attribution"
if docker inspect hermes >/dev/null 2>&1; then
  docker exec hermes sh -c "echo $MARK-container" >/dev/null 2>&1
  await 45 "SELECT count() FROM audit.host_exec WHERE arguments LIKE '%$MARK-container%' AND container_id != ''" 1 \
    "in-container exec carries a container id"
else
  note "no 'hermes' container on this host"
fi

step "3. File mutations"
TESTDIR=$(mktemp -d)
echo hello > "$TESTDIR/$MARK-create.txt"
mv "$TESTDIR/$MARK-create.txt" "$TESTDIR/$MARK-renamed.txt"
rm -f "$TESTDIR/$MARK-renamed.txt"
rmdir "$TESTDIR" 2>/dev/null || true
await 45 "SELECT count() FROM audit.host_file WHERE path LIKE '%$MARK-create%' AND op='open_write'" 1 \
  "file creation captured"
await 30 "SELECT count() FROM audit.host_file WHERE (path LIKE '%$MARK%' OR path_to LIKE '%$MARK%') AND op='rename'" 1 \
  "rename captured"
await 30 "SELECT count() FROM audit.host_file WHERE path LIKE '%$MARK-renamed%' AND op='unlink'" 1 \
  "delete captured"

step "4. Paths are absolute or honestly labelled"
unresolved=$(CH "SELECT count() FROM audit.host_file WHERE ts > now() - INTERVAL 10 MINUTE AND path_confidence='unresolved'" | tr -dc '0-9')
total=$(CH "SELECT count() FROM audit.host_file WHERE ts > now() - INTERVAL 10 MINUTE" | tr -dc '0-9')
if [ "${total:-0}" -gt 0 ]; then
  pct=$(( 100 * ${unresolved:-0} / total ))
  [ "$pct" -lt 10 ] && ok "only ${pct}% of recent file rows are unresolved paths" \
                    || bad "${pct}% of file rows have unresolved paths"
else
  note "no recent file rows to judge"
fi

step "5. Network"
curl -s -o /dev/null --max-time 15 https://example.com 2>/dev/null
await 45 "SELECT count() FROM audit.host_net WHERE direction='outbound' AND ts > now() - INTERVAL 3 MINUTE" 1 \
  "outbound connection captured"
await 30 "SELECT count() FROM audit.host_net WHERE direction='inbound' AND ts > now() - INTERVAL 30 MINUTE" 1 \
  "inbound connection captured"

step "6. Secret redaction"
# The AWS documentation example key. Nothing real is exposed by this test, and
# the point is that the string never reaches storage intact.
/bin/echo "$MARK-secret AKIAIOSFODNN7EXAMPLE" >/dev/null
sleep 12
leaked=$(CH "SELECT count() FROM audit.host_exec WHERE arguments LIKE '%AKIAIOSFODNN7EXAMPLE%'" | tr -dc '0-9')
masked=$(CH "SELECT count() FROM audit.host_exec WHERE arguments LIKE '%$MARK-secret%' AND redacted=1" | tr -dc '0-9')
[ "${leaked:-1}" -eq 0 ] && ok "no unmasked key reached the store" || bad "${leaked} rows contain the raw key"
[ "${masked:-0}" -ge 1 ] && ok "the event was stored, flagged redacted" || bad "redaction dropped the event instead of masking it"

step "7. Liveness beat"
await 90 "SELECT count() FROM audit.host_beat WHERE ts > now() - INTERVAL 3 MINUTE" 1 \
  "the shipper is emitting its liveness beat"

step "8. Shipper is keeping up with the sensor"
# Tetragon keeps only 5 x 10 MB of export. If Vector falls further behind than
# that, events are deleted before they are ever shipped -- silently. Lag is
# therefore a correctness measure, not a performance one.
newest=$(CH "SELECT toUnixTimestamp(max(ts)) FROM audit.host_exec" | tr -dc '0-9')
now=$(date +%s)
if [ -n "$newest" ] && [ "$newest" -gt 0 ]; then
  lag=$(( now - newest ))
  [ "$lag" -lt 120 ] && ok "newest stored event is ${lag}s old" || bad "shipper is ${lag}s behind"
else
  bad "no events in the store at all"
fi
# Scoped to the last 5 minutes. The full log carries every startup error the
# container ever had, so counting all of it measures the install's history
# rather than its current health.
errs=$(docker logs --since 5m audit-vector 2>&1 | grep -ciE '\bERROR\b' || true)
[ "${errs:-0}" -lt 5 ] && ok "vector log is clean (${errs:-0} recent error lines)" \
  || { bad "vector logged ${errs} errors in the last 5 minutes"
       docker logs --since 5m audit-vector 2>&1 | grep -E '\bERROR\b' | tail -2 | cut -c1-160 | sed 's/^/        /'; }

step "9. Parse failures are visible, not silent"
pf=$(CH "SELECT count() FROM audit.host_beat WHERE note LIKE 'PARSE FAILURE%' AND ts > now() - INTERVAL 1 HOUR" | tr -dc '0-9')
[ "${pf:-0}" -eq 0 ] && ok "no parse failures in the last hour" \
  || bad "${pf} events failed to parse -- a sensor output shape probably changed"

step "10. Retention is actually set"
# system.tables has no `ttl_expression` column (checked against 26.8); the TTL
# shows up in the CREATE statement, so that is what is matched.
nottl=$(CH "SELECT count() FROM system.tables WHERE database='audit' AND engine LIKE '%MergeTree%' AND create_table_query NOT LIKE '%TTL %'" | tr -dc '0-9')
[ "${nottl:-1}" -eq 0 ] && ok "every table has a TTL" \
  || bad "${nottl} tables have NO TTL and will grow without bound"

step "11. Least privilege is real"
w=$(docker exec audit-clickhouse clickhouse-client --user reader --password "$AUDIT_CLICKHOUSE_READER_PASSWORD" \
      --query "INSERT INTO audit.host_beat (ts,host,role,vector_version,note) VALUES (now(),'x','x','x','x')" 2>&1)
echo "$w" | grep -qiE 'denied|not enough|readonly|cannot' && ok "reader cannot write" || bad "READER CAN WRITE: $w"
r=$(docker exec audit-clickhouse clickhouse-client --user ingest --password "$AUDIT_CLICKHOUSE_INGEST_PASSWORD" \
      --query "SELECT count() FROM audit.host_exec" 2>&1)
echo "$r" | grep -qiE 'denied|not enough|cannot' && ok "ingest cannot read" || bad "INGEST CAN READ: $r"

step "Volume baseline (for docs/AUDIT_PLAN.md §8)"
CH "SELECT concat('  ', name, ': ', toString(total_rows), ' rows, ', formatReadableSize(total_bytes)) FROM system.tables WHERE database='audit' AND total_rows > 0 ORDER BY total_rows DESC"
CH "SELECT concat('  bytes/row: ', toString(round(sum(total_bytes)/greatest(sum(total_rows),1),1))) FROM system.tables WHERE database='audit'"

printf '\n\033[1m%d passed, %d failed, %d skipped\033[0m\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
