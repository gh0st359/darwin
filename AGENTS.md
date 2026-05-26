# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Project Darwin is a pure-Python symbolic/causal AI kernel. Zero external runtime dependencies — stdlib only. See `README.md` for full architecture details and CLI reference.

### Running services

The primary development workflow uses two processes (see `wiki/Two-Terminal-Workflow.md`):

1. **Brain daemon**: `darwin brain` (v3) or `darwin brain --kernel v4 --memory <path>` (v4)
2. **Chat client**: `darwin connect` (connects to the brain daemon via TCP on port 9870 by default)

Use `--port <N>` on both commands to avoid collisions if port 9870 is busy. Shut down the brain from the chat client with `/shutdown-brain`.

### Tests

```
python3 -m unittest discover -s tests
```

69 tests, runs in ~7 seconds, no network or GPU needed. Use `python3` (not `python`) — the VM may not have `python` aliased.

### Gotchas

- The `darwin` CLI is installed to `~/.local/bin/`. Ensure `PATH` includes it: `export PATH="$HOME/.local/bin:$PATH"`.
- There is no linter or formatter configured in this repo. The project enforces code style via review, not tooling.
- The brain daemon writes `darwin_memory.sqlite3`, `darwin_runtime_state.json`, and `training_logs/` to the current directory. Use `--memory /tmp/<file>.sqlite3` for ephemeral experiments to avoid polluting the repo checkout.
- Corpus ingestion (`darwin ingest-corpus`) is an offline CLI step that must run before `darwin brain --kernel v4` if you want knowledge atoms and generated worlds.
