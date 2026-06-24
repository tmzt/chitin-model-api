# @chitin/model-api

Node.js bindings for [`chitin-model-api`](../model_api_server) — the
Unix-domain-socket inference server that owns a llama.cpp `Session`
plus a smart KV cache. Connect from JS, send a prompt, get a
response.

```js
const { Client } = require('@chitin/model-api');

const c = await Client.connect('/tmp/chitin-model-api.sock');
const r = await c.inference({
  role: 'deep',
  input: 'Write a haiku about Rust.',
  maxTokens: 64,
});
console.log(r.text);
await c.shutdown();
```

The server is a separate process. Start it first (see
[Running the server](#running-the-server) below); the bindings just
talk to whichever socket path you pass to `connect`.

## Install

> Today this crate is in-tree. Once published it will install as a
> normal npm package with prebuilt binaries via napi-rs.

```bash
cd model_api_node
npm install
npm run build       # release build → model-api.node
node test.js        # smoke test against a running server
```

`npm run build` invokes `@napi-rs/cli` which calls `cargo build
--release` under the hood and copies the resulting `.node` file into
the package root. Targets are listed in `package.json#napi.triples`
(macOS aarch64 / x86_64, Linux aarch64 / x86_64).

## API

### `Client.connect(socketPath: string): Promise<Client>`

Open a UDS connection and perform the protocol Hello handshake.
Resolves with a `Client`; rejects if the socket isn't reachable, the
server replies with an `InferenceError`, or the protocol version
doesn't match.

### `client.modelName: string`

Model name the server reported in its Hello (e.g.
`"gemma-4-26B-A4B-it-UD-Q4_K_M"`).

### `client.gpuMemoryMb: number | null`

Best-effort GPU memory in megabytes. `null` for CPU-only or
unknown.

### `client.protocolVersion: number`

Wire protocol version. Use it to gate client behaviour against
future server bumps.

### `client.inference(req: InferenceRequest): Promise<InferenceResponse>`

Submit one inference request. The server serializes requests
behind a single slot — concurrent calls from your code will queue
on the underlying mutex. Resolves with the final response;
rejects on `InferenceError` from the server or transport failure.

```ts
interface InferenceRequest {
  /** Routing role on the server side. */
  role: 'fast' | 'deep' | 'asr' | 'omni'

  /** Prompt text. */
  input: string

  /** Cap on tokens generated for this turn. */
  maxTokens: number

  /**
   * Session id for the smart KV cache.
   *  - undefined → stateless one-shot
   *  - ''        → create a fresh persistent session
   *  - 's-…'     → resume / continue session 's-…'
   */
  sessionId?: string

  /**
   * Caller-supplied Merkle root of the conversation they think the
   * server's KV cache already represents. Server falls through to a
   * full rebuild on mismatch. Default 0 = "I don't track one".
   */
  cacheHash?: bigint

  /**
   * Skip prefilling the model's `<think>\n` chain-of-thought prefix.
   * Useful for plain chat-completion APIs that don't want a think
   * block in the output. Default false.
   */
  disableThinkPrefix?: boolean
}

interface InferenceResponse {
  /** Cleaned answer text (think blocks already stripped). */
  text: string

  /** Server-assigned session id if a session was created/continued. */
  sessionId?: string

  /**
   * Raw model output, including `<think>...</think>` and any
   * injected `<tool_response>...</tool_response>` blocks. Use this
   * for transcript logging if you care about reasoning traces.
   */
  rawText?: string

  /**
   * Tool-response payloads the dispatcher spliced in during
   * generation, in order. Empty when no injections fired.
   */
  injections: string[]
}
```

### `client.shutdown(): Promise<void>`

Ask the server to drain its queue and exit. Resolves once the
server replies with its `Goodbye` (or on a clean disconnect). The
client's socket closes when the `Client` is garbage-collected.

## Running the server

The bindings only talk to a socket; you start the server separately
from the workspace root:

```bash
# Full backend (loads a GGUF model, requires ~16GB GPU/CPU memory):
make run-ma                                                     \
  MA_SOCKET=/tmp/chitin-model-api.sock                          \
  MA_MODEL=~/path/to/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf

# Stub backend (echoes prompts back, no model load — for the
# Node smoke test or unit tests):
cargo build -p model_api_server --bin chitin-model-api --no-default-features
/tmp/cm-target/debug/chitin-model-api --socket /tmp/chitin-model-api.sock
```

Then on the JS side:

```bash
node test.js /tmp/chitin-model-api.sock
```

## Threading model

Each method wraps the underlying sync Rust client in
`tokio::task::spawn_blocking`, so the Node event loop never blocks
on UDS I/O. The Rust side holds a single `Mutex<UnixStream>` per
`Client`; concurrent JS calls serialize at that mutex (which is fine
— the server's slot is also single-threaded).

If you want parallel inflight requests from JS, open multiple
`Client`s; each gets its own queue position with the server.

## Streaming + progress (not yet wired)

The wire protocol carries `Chunk` and `Progress` frames for
streaming token output and real-time progress events; the Node
bindings currently drain them silently while waiting for
`InferenceComplete`. Surfacing them as a callback or
`AsyncIterator` is a follow-up — search for the `TODO` notes in
`src/lib.rs` once you want it.

## Wire protocol

Length-prefixed bincode, `[u32 LE len][bincode payload]`, capped
at 64 MB per frame. Message envelopes are defined in
[`model_api_proto`](../model_api_proto/src/lib.rs); the bindings
mirror only the request/response halves (request senders + Arc
dispatchers stay Rust-side).

## License

MIT OR Apache-2.0.
