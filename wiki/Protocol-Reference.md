# Brain Daemon Protocol

`darwin brain` exposes Darwin over a plain TCP socket with a JSON-lines
protocol. One JSON object per line, terminated by `\n`. Default
endpoint is `127.0.0.1:9870`.

You can use the protocol directly from any language, or use the
provided `DarwinClient` (`darwin.server.DarwinClient`) from Python.

## Framing

- Each direction: one JSON object per line, UTF-8, terminated by `\n`.
- The server multiplexes responses by `id`. Requests carrying an `id`
  receive a response carrying the same `id`. Requests without an `id`
  receive a response without an `id` (or no response at all).
- Background events are always pushed asynchronously; they never carry
  an `id`. They are only sent to subscribers that have called
  `subscribe`.

## Lifecycle

1. Open TCP connection.
2. Optionally `{"cmd": "subscribe"}` to opt in to the event firehose.
3. Send chat / command messages.
4. Optionally `{"cmd": "unsubscribe"}` to stop receiving events.
5. Close the connection (TCP FIN) when done. The brain keeps running.

## Client → Server messages

### `chat`
```json
{"cmd": "chat", "id": "42", "message": "What do you believe about open_curtains?"}
```
Drives one full cognitive cycle and returns Darwin's response.

### `command`
```json
{"cmd": "command", "id": "43", "command": "/beliefs"}
```
Runs any chat-window slash command on the brain and returns its output
as `lines`. See [CLI Reference](CLI-Reference.md) for the full list.

### `subscribe` / `unsubscribe`
```json
{"cmd": "subscribe",   "id": "44"}
{"cmd": "unsubscribe", "id": "45"}
```
Opt the connection in or out of the background event firehose.
Unsubscribed clients receive zero events — this is what makes the
default `darwin connect` chat window clean.

### `ping`
```json
{"cmd": "ping", "id": "46"}
```
Returns `{"type": "pong", "id": "46", "ts": ...}`.

### `shutdown`
```json
{"cmd": "shutdown", "id": "47"}
```
Asks the brain to stop. The server acknowledges, then exits.

## Server → Client messages

### `welcome` (sent on connect)
```json
{
  "type": "welcome",
  "brain": "darwin",
  "loops": ["experiment", "simulation", "dream", "self_modification", "uncertainty"],
  "running": true
}
```

### `response` (reply to `chat`)
```json
{
  "type": "response",
  "id": "42",
  "text": "The beliefs I can defend are ...",
  "plan": {
    "plan_id": "…",
    "mode": "belief_answer",
    "thesis": "…",
    "answer_points": ["…"],
    "causal_claims": [
      {"action": "open_curtains", "variable": "room_bright",
       "effect": "False -> True", "confidence": 0.86, "samples": 4,
       "condition": "always"}
    ],
    "uncertainty_levels": [
      {"target": "answer", "level": 0.4, "reason": "grounded memory was thin"}
    ],
    "referenced_experiences": [...],
    "self_reflection": [...],
    "tone": "neutral",
    "target_length": "medium",
    "confidence": 0.6
  }
}
```

### `command_result` (reply to `command`)
```json
{"type": "command_result", "id": "43", "lines": ["- if always: open_curtains -> room_bright …"]}
```

### `event` (broadcast to subscribers only)
```json
{
  "type": "event",
  "kind": "simulation",
  "loop": "simulation",
  "content": "Mental simulation: step 1: open_curtains (conf 0.33, unc 0.67) -> …",
  "payload": {"chain": {...}},
  "timestamp": 1758...
}
```

`kind` is one of: `experiment`, `simulation`, `dream`,
`self_modification`, `uncertainty`, `reflection`, `thought`,
`runtime`, `error`. `loop` is the producing loop name.

### `subscribed` / `unsubscribed`
```json
{"type": "subscribed",   "id": "44"}
{"type": "unsubscribed", "id": "45"}
```

### `pong`
```json
{"type": "pong", "id": "46", "ts": 1758...}
```

### `shutting_down`
```json
{"type": "shutting_down", "id": "47"}
```

### `error`
```json
{"type": "error", "id": "...", "message": "…"}
```

## Backpressure

Each client has a bounded outbound queue (size 512). If a client is so
slow the queue fills, the daemon drops the **oldest** queued message
to make room. This keeps a slow client from stalling the background
cognition loops.

## Using `DarwinClient` from Python

```python
from darwin.server import DarwinClient

client = DarwinClient(host="127.0.0.1", port=9870)
events = []
client.connect(events.append)

# Subscribe to the firehose (optional)
client.subscribe_events()

# Chat
result = client.chat("What do you believe about open_curtains?")
print(result["text"])

# Run a slash command
for line in client.command("/beliefs"):
    print(line)

client.close()
```

## Security note

The default bind is `127.0.0.1`. Do not expose the brain on a public
network without putting authentication in front of it — there is no
auth in the v2 protocol. For multi-machine setups, use an SSH tunnel
or a reverse proxy with TLS + auth.
