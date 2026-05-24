# The Two-Terminal Workflow

The whole point of v2 is that Darwin's mind keeps running whether or
not you are talking to it. To make that visible (and to keep your
chat window clean), Darwin runs as a daemon you attach to.

## Terminal 1: the brain

```bash
darwin brain
```

This process owns Darwin. It runs the five background loops, holds the
SQLite memory open, and listens on TCP for chat clients. Output in
this window is the *firehose*: every experiment fired, every mental
simulation, every dream, every self-modification proposal, every
uncertainty scan, every thought generated when a chat client sends a
message.

Useful flags:

```bash
darwin brain --interval 2.0          # base interval for fast loops (sec)
darwin brain --port 9870             # TCP port (default 9870)
darwin brain --host 127.0.0.1        # bind interface (loopback by default)
darwin brain --memory ~/state.sqlite3
darwin brain --dlm gemma             # optional gemma-3-270m renderer
darwin brain --quiet                 # suppress brain-window output
```

The `--interval` controls the *experiment* loop and proportionally
scales the others (simulation = 1.5x, dream = 4x, self_modification =
6x, uncertainty = 3x). Lower it for a busier mind; raise it for a
calmer one.

To stop: Ctrl-C. The brain checkpoints `darwin_runtime_state.json`
and the SQLite file on shutdown.

## Terminal 2: a clean chat client

```bash
darwin connect
```

You see:

```
Connected to brain at 127.0.0.1:9870
Clean chat window. Background thinking streams in the 'darwin brain' terminal.
Type your messages, or /help for commands. /exit to leave the chat (brain keeps running).
you> 
```

Type a message. Darwin replies with `darwin>`. That is the entire
interface. Background events never leak into this window — the chat
client does not even subscribe to them at the protocol level.

To leave the chat without stopping the brain: `/exit` or Ctrl-D.

To stop the brain from a client: `/shutdown-brain`.

## Mirroring the firehose into the chat window (opt-in)

If you want a single window that shows both:

```bash
darwin connect --watch-events
```

In this mode the client sends `{"cmd": "subscribe"}` once on startup
and the brain streams events to it. Events appear above the `you>`
prompt with ANSI line-erase so they do not chop your typed line.

You can also have a chat window and a separate "watch" window:

```bash
darwin connect                 # window 2: clean chat
darwin connect --watch-events  # window 3: chat + event mirror
```

## Multiple clients

Any number of `darwin connect` sessions can attach to the same brain at
once. They all share one Darwin, one memory, one continuous learning
trajectory. Anything one client teaches Darwin, the other clients
immediately benefit from. Each subscription is independent — one
window can `--watch-events` while another stays silent.

## Why two terminals

- The chat REPL is for conversation. Conversations need focus.
- The brain stream is for inspection. You want to see Darwin
  thinking, but not while you are mid-sentence.
- Two windows acknowledge that those are two different jobs.

When you do not need either, just close them. The brain process
keeps running in Terminal 1 regardless.
