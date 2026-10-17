# Event stream restart and recovery contract

`/api/plugins/astation/ws/events` (standalone: `/ws/events`) uses a **broadcaster epoch**, not a durable/global sequence. Every newly constructed broadcaster has a fresh opaque `stream_epoch` UUID. Ordinary WebSocket reconnects to the same broadcaster retain it.

Client resume: `?after=<last-seq>&epoch=<stream_epoch>`.

The first `stream.ready` envelope has `seq: 0`, top-level `stream_epoch`, and payload:

- `stream_epoch`: same opaque identifier
- `last_seq`: current high-water mark, diagnostic only (do not advance the client cursor to it)
- `replay`: true only if the supplied epoch matches and retained history covers the requested cursor
- `cursor_reset`: true when a supplied cursor cannot be honored (missing/wrong epoch, cursor ahead of server, or history gap)
- `connection_generation`: existing upstream socket-generation field; not a stream epoch

When `cursor_reset` is true, the server ignores the old cursor, sends `replay:false`, and subscribes as fresh (the existing `stream.reset` control follows). The client must atomically adopt the new epoch, reset BOTH transport and transcript sequence high-water marks, and reload authoritative exact-profile history while buffering/applying new events. Do not delete canonical rows or visible identities during this reconciliation. Treat seq=0 controls before sequence deduplication. Reject delayed frames from an old epoch. Every numbered frame carries top-level `stream_epoch`; no sequence is comparable across epochs.

Same-epoch, covered reconnects replay normally without resetting sequence deduplication. Legacy `after` without `epoch` is deliberately not accepted as proof of continuity. A fresh connection without `after` has `cursor_reset:false`, `replay:false`; existing initial reload/reset behavior still applies.

## Durable identity recovery

Hosted capture resolves ownership from the serving Hermes registry, its canonical stored-session lookup, and canonical profile-home helpers. The transport/default profile and event-provided profile claims are not ownership proofs. Stored IDs alone can collide across profiles and are not sufficient. A registry session created after the frame's original receipt cannot claim that frame.

The private file-backed inbox is separate from original chat databases, permission 0600, bounded by configured row/byte quotas, and paginated. A verified identity is persisted before canonical capture, allowing crash/restart recovery without consulting a new process's reused live handle. Capture/ack retries use a durable event identity for both chat rows and run events. Unresolved frames are retried by a timer even without subsequent traffic.

A frame that never had any provable owner before its entire host registry disappeared cannot safely be automatically assigned after restart. It remains private on disk; health exposes pending/old-epoch counts, not payloads. Such rows require explicitly authorized ownership review. Quota/storage/capture failures produce metadata-only reset signals, never silently attribute to the default profile. No global history merge or original split-row repair is performed.

Plugin health reports `loaded_revision` captured at module import. Installation on disk is not activation; restart must be performed externally only after active work, including parent/subagent workers, is safe to stop.
