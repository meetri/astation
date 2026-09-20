-- The audit store's schema. Applied by install.sh on every run, so every
-- statement is CREATE ... IF NOT EXISTS and re-running is a no-op.
--
-- Shape decisions that apply to every table:
--
--   PARTITION BY toDate(ts)   A day is the unit retention works in, so TTL
--                             drops whole partitions instead of rewriting
--                             parts. Dropping 14-day-old file events must not
--                             cost a merge of everything newer.
--
--   ORDER BY (host, ts)       Every question is asked about a host over a time
--                             range. The sorting key is also the primary index
--                             in MergeTree, so this is what makes "what
--                             happened on that box between 14:00 and 14:05" a
--                             range scan rather than a full read.
--
--   LowCardinality(...)       For columns with a small fixed vocabulary. It is
--                             a dictionary encoding, and on this data (a
--                             handful of hosts, ~6 operations, a few container
--                             names) it is most of why the store is small.
--
--   No `raw` column           The raw NDJSON is in S3 and is the record of
--                             record. Keeping a second copy here would roughly
--                             double the hot store to re-answer a question S3
--                             already answers.
--
--   TTLs are NOT written here. They come from audit-setup/retention.yaml via
--   scripts/audit_retention.py, because retention has to be decided in exactly
--   one place and this file is not it. A table created here has no TTL until
--   the renderer applies one -- install.sh does that as its last step.

CREATE DATABASE IF NOT EXISTS audit;

-- Every process that ran. The spine: host_file and host_net rows carry an
-- exec_id that joins here, and exec_id -> parent_exec_id walks the tree back
-- to whatever started it.
CREATE TABLE IF NOT EXISTS audit.host_exec
(
    ts                DateTime64(3, 'UTC'),
    host              LowCardinality(String),
    exec_id           String,
    parent_exec_id    String,
    pid               UInt32,
    tid               UInt32,
    uid               UInt32,
    -- The login uid, inherited across su/sudo and setuid, so it survives the
    -- privilege change that uid does not. 4294967295 means "not set", which is
    -- normal inside a container.
    auid              Int64,
    binary            String,
    arguments         String,
    cwd               String,
    flags             String,
    container_id      String,
    container_name    LowCardinality(String),
    parent_exec_id_2  String,
    parent_binary     String,
    parent_arguments  String,
    parent_pid        UInt32,
    in_init_tree      UInt8,
    -- 1 when the shipper masked something key-shaped before storing it.
    redacted          UInt8 DEFAULT 0
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (host, ts);

CREATE TABLE IF NOT EXISTS audit.host_exit
(
    ts              DateTime64(3, 'UTC'),
    host            LowCardinality(String),
    exec_id         String,
    pid             UInt32,
    uid             UInt32,
    binary          String,
    container_id    String,
    container_name  LowCardinality(String),
    status          Int32,
    -- Tetragon reports this only sometimes; 0 means "not reported", not
    -- "killed by signal 0".
    signal          LowCardinality(String)
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (host, ts);

-- What changed on disk. See audit-setup/tetragon/policies/file-mutations.yaml for
-- why `op` is open-time rather than write-time, and what path_confidence means.
CREATE TABLE IF NOT EXISTS audit.host_file
(
    ts                DateTime64(3, 'UTC'),
    host              LowCardinality(String),
    exec_id           String,
    pid               UInt32,
    uid               UInt32,
    binary            String,
    container_id      String,
    container_name    LowCardinality(String),
    -- open_write | unlink | rename | write
    op                LowCardinality(String),
    path              String,
    -- The destination, for rename. Empty otherwise.
    path_to           String,
    -- absolute     the caller passed a full path; this is exactly right.
    -- cwd_resolved  we joined a relative path to the process's recorded cwd.
    --               Right unless the process changed directory after exec.
    -- resolved      the kernel gave us the path (LSM hook); always right.
    path_confidence   LowCardinality(String),
    cwd               String,
    -- Which kernel hook produced the row. Two hooks see writes on purpose;
    -- keeping the source makes their disagreement visible instead of confusing.
    source_hook       LowCardinality(String),
    redacted          UInt8 DEFAULT 0
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (host, ts);

-- Who talked to whom. Loopback is excluded at the sensor.
CREATE TABLE IF NOT EXISTS audit.host_net
(
    ts              DateTime64(3, 'UTC'),
    host            LowCardinality(String),
    exec_id         String,
    pid             UInt32,
    uid             UInt32,
    binary          String,
    container_id    String,
    container_name  LowCardinality(String),
    -- outbound | inbound | close
    direction       LowCardinality(String),
    -- Strings, not IPv4/IPv6 columns: the same socket can be reported as
    -- `::ffff:172.17.0.5` or `172.17.0.5` depending on the family, and a typed
    -- column would reject one of them. Normalisation belongs in a view, not in
    -- a constraint that drops rows.
    saddr           String,
    sport           UInt32,
    daddr           String,
    dport           UInt32,
    family          LowCardinality(String)
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (host, ts);

-- Logins and privilege changes, parsed from journald.
CREATE TABLE IF NOT EXISTS audit.host_auth
(
    ts          DateTime64(3, 'UTC'),
    host        LowCardinality(String),
    unit        LowCardinality(String),
    -- accepted | failed | session_open | session_close | sudo | other
    kind        LowCardinality(String),
    user        String,
    source_ip   String,
    message     String
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (host, ts);

-- Container lifecycle, from the docker event stream.
CREATE TABLE IF NOT EXISTS audit.host_container
(
    ts              DateTime64(3, 'UTC'),
    host            LowCardinality(String),
    action          LowCardinality(String),
    container_id    String,
    container_name  LowCardinality(String),
    image           String,
    message         String
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (host, ts);

-- The liveness class. One row a minute from every host, whatever else is
-- happening. Its ABSENCE is the alarm, which is why it must not depend on the
-- host doing anything: a quiet host and a dead sensor look identical without it.
CREATE TABLE IF NOT EXISTS audit.host_beat
(
    ts              DateTime64(3, 'UTC'),
    host            LowCardinality(String),
    role            LowCardinality(String),
    vector_version  LowCardinality(String),
    note            String
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (host, ts);

-- ---------------------------------------------------------------------------
-- Phase 2 classes. Created now, empty until the astation plugin ships to them,
-- so that adding the session layer is a shipper change and not a migration.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS audit.agent_tool
(
    ts                 DateTime64(3, 'UTC'),
    host               LowCardinality(String),
    profile            LowCardinality(String),
    stored_session_id  String,
    turn_id            String,
    tool_call_id       String,
    tool_name          LowCardinality(String),
    args               String,
    -- The shell command this tool was about to run, when there is one. THE
    -- JOIN KEY: a `host_exec` row's argv contains this text, in the same
    -- container, moments later. Kept apart from `args` so the attribution
    -- query never has to parse a summary.
    command            String,
    result_hash        String,
    blocked            LowCardinality(String),
    -- How long the tool took, from `post_tool_call`. Only ever set on a
    -- `phase = 'result'` row: `pre_tool_call` fires BEFORE the work, so the
    -- call row cannot know it.
    duration_ms        UInt32,
    -- Which half of the call this row is. `call` is written before the tool
    -- runs and is the one that must exist no matter what -- a tool that hangs
    -- or is killed never reaches `post_tool_call`, and a call that vanished
    -- from the record because it never returned is the worst possible gap in
    -- an audit trail. `result` is written after and carries only the outcome:
    -- how long, whether it succeeded, and the error CLASS if it did not.
    -- Deliberately never the tool's output (owner, 2026-09-20): a file read
    -- returns the file, and the archive cannot be edited or deleted for the
    -- retention window.
    -- Empty means a row written before this column existed; those are calls.
    phase              LowCardinality(String) DEFAULT '',
    status             LowCardinality(String) DEFAULT '',
    error_type         LowCardinality(String) DEFAULT '',
    redacted           UInt8 DEFAULT 0
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (host, ts);

CREATE TABLE IF NOT EXISTS audit.agent_llm
(
    ts                 DateTime64(3, 'UTC'),
    host               LowCardinality(String),
    profile            LowCardinality(String),
    stored_session_id  String,
    turn_id            String,
    model              LowCardinality(String),
    prompt_hash        String,
    response_hash      String,
    input_tokens       UInt32,
    output_tokens      UInt32,
    cache_read_tokens  UInt32,
    cache_write_tokens UInt32
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (host, ts);

CREATE TABLE IF NOT EXISTS audit.agent_session
(
    ts                 DateTime64(3, 'UTC'),
    host               LowCardinality(String),
    profile            LowCardinality(String),
    stored_session_id  String,
    event              LowCardinality(String),
    cwd                String,
    detail             String
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (host, ts);

-- Who asked this store what. The analyst is audited by the thing it analyses,
-- which is why this class has the longest hot window in retention.yaml.
CREATE TABLE IF NOT EXISTS audit.audit_query
(
    ts          DateTime64(3, 'UTC'),
    host        LowCardinality(String),
    -- `app:<user>` or `agent:<profile>`
    caller      String,
    route       LowCardinality(String),
    sql         String,
    rows_out    UInt64,
    duration_ms UInt32,
    error       String
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (host, ts);


-- Columns added after a store was first created. `CREATE TABLE IF NOT EXISTS`
-- above is a no-op on an existing table, so a new column needs its own
-- idempotent ALTER or it silently never arrives. Every future column goes here
-- too, not only in the CREATE.
ALTER TABLE audit.agent_tool ADD COLUMN IF NOT EXISTS command String;
ALTER TABLE audit.agent_tool
    ADD COLUMN IF NOT EXISTS phase LowCardinality(String) DEFAULT '';
ALTER TABLE audit.agent_tool
    ADD COLUMN IF NOT EXISTS status LowCardinality(String) DEFAULT '';
ALTER TABLE audit.agent_tool
    ADD COLUMN IF NOT EXISTS error_type LowCardinality(String) DEFAULT '';

-- ---------------------------------------------------------------------------
-- Attribution: which SESSION caused a process to run.
--
-- A view rather than a column on `host_exec`, deliberately. The kernel's rows
-- are what happened and are never rewritten; the agent's rows are what it
-- asked for. Joining them at read time keeps the two claims separable, so a
-- compromised agent can lie in its own channel without touching the other.
--
-- The match is: same container, the tool's command appearing in the process's
-- argv, and the exec happening within 15 seconds AFTER the tool call. A
-- process that matches nothing is still returned, with an empty session --
-- "ran without a tool call" is itself worth seeing.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW audit.exec_attributed AS
SELECT
    e.ts               AS ts,
    e.host             AS host,
    e.exec_id          AS exec_id,
    e.pid              AS pid,
    e.uid              AS uid,
    e.binary           AS binary,
    e.arguments        AS arguments,
    e.container_id     AS container_id,
    t.stored_session_id AS stored_session_id,
    t.turn_id          AS turn_id,
    t.tool_call_id     AS tool_call_id,
    t.tool_name        AS tool_name,
    t.profile          AS profile,
    if(t.stored_session_id != '', 'tool_call', 'none') AS attribution
FROM audit.host_exec AS e
LEFT JOIN audit.agent_tool AS t
    ON  t.command != ''
    AND position(e.arguments, t.command) > 0
    AND t.ts <= e.ts
    AND e.ts <= t.ts + INTERVAL 15 SECOND;
