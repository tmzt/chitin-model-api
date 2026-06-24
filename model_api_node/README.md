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
type ToolMode = 'auto' | 'server' | 'client'
type JsonMode = 'none' | 'tool-only' | 'thinking-with-tools'
              | 'any-json' | 'notes-classifier-json'

interface ToolDef {
  name: string
  description: string
  parameters: Array<{
    name: string
    type: string        // "string" | "integer" | "boolean" | "number" | "array" | "object"
    required: boolean
    description: string
  }>
}
interface ToolCall { id: string; name: string; argumentsJson: string }
interface ToolResult { callId: string; output: string }

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
   * Caller-supplied Merkle root of the conversation. 0n / omit = "I
   * don't track one".
   */
  cacheHash?: bigint

  // ── Tool wire ──
  /**
   * Tools the model can call this turn. Model-agnostic shape; the
   * server hands these to the loaded model's format adapter so
   * Gemma 4 / ChatML / etc. get a model-native system prompt.
   * Empty / omit for plain chat.
   */
  tools?: ToolDef[]

  /**
   * Results from tool calls the client executed in response to a
   * prior turn. The server formats each via the model's
   * `ToolFormat::format_response` and folds them into the next user
   * turn before generation.
   */
  toolResults?: ToolResult[]

  /**
   * Where tool calls are dispatched.
   *  - 'auto'   — 'client' when tools are present, 'server' otherwise.
   *               Default.
   *  - 'server' — in-band dispatcher runs server-known tools
   *               (memory, project graph, etc.) mid-generation. The
   *               client never sees the calls; the response shows
   *               them in `injections`.
   *  - 'client' — server buffers tool calls and returns them in
   *               `toolCalls`. Client runs the tool externally and
   *               replies with `toolResults` next turn. Matches the
   *               OpenAI function-calling loop — what agent clients
   *               (pi-ai, etc.) expect.
   */
  toolMode?: ToolMode

  // ── Sampler / output shape ──
  temperature?: number
  topP?: number
  repPenalty?: number
  presencePenalty?: number
  /** Per-request cap; overrides `maxTokens` when smaller. */
  maxTokensOverride?: number
  /** Override the session/role system prompt for this turn. */
  systemPrompt?: string
  /** Default 'none' (free-form). */
  jsonMode?: JsonMode
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
   * Server-side tool dispatches that fired during generation, in
   * order. Empty when no injections fired or when `toolMode` was
   * `'client'`.
   */
  injections: string[]

  /**
   * Tool calls the model emitted that the client needs to execute.
   * Populated when `toolMode` was `'client'` (or `'auto'` with
   * tools present). Empty otherwise.
   */
  toolCalls: ToolCall[]
}
```

## Agent loop example

```js
const { Client } = require('@chitin/model-api');

const calc = (expr) => String(eval(expr));   // toy tool

const c = await Client.connect('/tmp/chitin-model-api.sock');
const sessionId = `agent-${Date.now()}`;
let toolResults = [];
let input = 'What is (2+2)*7?';

for (let turn = 0; turn < 5; turn++) {
  const r = await c.inference({
    role: 'deep',
    input,
    maxTokens: 256,
    sessionId,
    tools: [{
      name: 'calculator',
      description: 'Evaluate a math expression',
      parameters: [{ name: 'expr', type: 'string', required: true, description: '' }],
    }],
    toolResults,
  });

  if (r.toolCalls.length === 0) {
    console.log('Final answer:', r.text);
    break;
  }

  // Model wants tools called — run them, queue results for next turn.
  toolResults = r.toolCalls.map((tc) => {
    const args = JSON.parse(tc.argumentsJson);
    return { callId: tc.id, output: calc(args.expr) };
  });
  input = '';   // model has the prior turn via the session
}

await c.shutdown();
```

### `client.shutdown(): Promise<void>`

Close this client's connection. Does **not** shut down the server —
just decrements the slot's reference count; other clients (and the
server process) continue. Resolves once the server replies with its
`Goodbye` (or on a clean disconnect). Equivalent to letting the
`Client` go out of scope; `shutdown` is the polite version that lets
the server log a clean hangup instead of an abrupt EOF.

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
