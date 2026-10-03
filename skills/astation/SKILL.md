---
name: astation
description: Use when working through Agent Station's mobile UI. Keep work in the workspace, present it for mobile review, and register important artifacts.
---

# Agent Station mobile research output

Apply these rules when the user is working through the Agent Station client.
They make work visible, reviewable, and openable from the iPhone UI.

## Workspace boundary

1. Treat the session's current workspace (normally the current working
   directory supplied when the project session was created) as the write
   boundary.
2. Keep every file you intentionally create or update inside that current
   workspace. Do not put work products, helper scripts, logs, downloaded data,
   or intermediate results elsewhere.
3. Put disposable scripts, transient data, command output, and other scratch
   material under `<workspace>/tmp/`; never `/tmp`, `/var/tmp`, or another
   directory outside the workspace. Create `<workspace>/tmp/` if needed.
4. Runtime-managed caches and state that tools create internally are outside
   this filing rule; do not deliberately place research work in them.

## Paths in replies

- Whenever the user should open, inspect, or reuse a file, give its absolute
  path. Never give only a relative path.
- Put an important path on its own line when practical, with a short label on
  the preceding line. This makes it easy for the mobile client to recognize.
- A path shown in prose is a reference. It is not a substitute for explicitly
  registering an important deliverable.

## Markdown for the mobile UI

- Use ordinary Markdown headings, bullets, short paragraphs, and compact
  tables when they improve scanning.
- Put code in triple-backtick, language-labelled fenced code blocks such as
  `python`, `rust`, `bash`, `json`, or `text`. This enables syntax-coloured
  rendering. Do not paste substantial code as unformatted prose.
- Keep the chat response concise. Put long reports, logs, tables, and source
  listings in workspace files, then provide their absolute paths.

## Register important artifacts

Agent Station detects changed workspace files automatically. For any important
user-facing output—report, figure, audio, video, dataset, rendered document,
metrics file, or useful script—also make the delivery intent explicit:

1. Verify that the file exists at the expected absolute path and is complete.
2. Emit exactly one line for it in the final response:

```text
MEDIA:/absolute/path/to/file
```

The `MEDIA:` token must begin the line and the path must be absolute. Agent
Station uses that line to ingest/register the artifact and render an artifact
chip. Use one line per artifact. Do not put multiple paths on one `MEDIA:`
line.

Do not emit `MEDIA:` for a nonexistent path, a disposable scratch file, a
source file merely mentioned for reference, or every file touched during a
change. Register the outputs that the user is likely to open, listen to, view,
download, compare, or preserve.

## File durable work by purpose

Within the workspace, use these folders for outputs worth keeping:

- `figures/` — plots, diagrams, and images
- `results/` — measurements, metrics, and tables
- `reports/` — write-ups meant to be read
- `data/` — datasets and derived data
- `notes/` — working notes

When useful, put an experiment folder above the role folder, for example
`clock-isolation-v1/results/metrics.json`. Agent Station derives artifact tags
from this layout when the file is ingested.

## Completion check

Before finishing a turn that created files, check that:

- all intentional writes stayed inside the current workspace;
- scratch material is under `<workspace>/tmp/`;
- user-facing file references are absolute paths;
- important deliverables exist and each has a line-anchored `MEDIA:` entry;
- code shown inline uses a language-labelled fence;
- the reply distinguishes durable outputs from scratch files.
