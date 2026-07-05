# chitin-model-api

The model-facing side of chitin: a UDS server that owns the local GPU/accelerator, and clients that talk to it over a length-prefixed bincode wire.

Four crates:

- **`model_api_proto`** — Wire types (`InferenceRequest`, `ToolCall`, `StreamEvent`, `Turn`, `SessionMode`). Bincode-serialisable, no runtime deps beyond serde.
- **`model_api_server`** — UDS server hosting one `Slot` backend. Two backends today: `llama-cpp` (via `llama_engine`) and `litert-lm` (Google LiteRT-LM v0.13.1, session-keyed `Conversation` pool for multi-turn KV reuse). Ships as the `chitin-model-api` binary.
- **`model_api_client`** — Sync + async client. `SyncClient` is the shape napi/JNI bindings want; async is used by workspace consumers on the smol runtime.
- **`model_api_node`** — napi-rs binding exposing `SyncClient` to Node.js.

## Tool calling

Two modes on the wire:

- `ToolMode::Server` — the server owns tool dispatch. Model output flows through `tool_text::extract_tool_calls` internally; the client only ever sees final text.
- `ToolMode::Client` — the server surfaces raw `tool_calls` on the response for the client to dispatch. Supports memory tools, MCP-backed workspace tools (Gmail / Calendar / Drive), and anything else the caller has bridged.

`ToolResult`s go back in the next request; the server threads them through the chat template so the model sees them on the second turn.

## Where this lives

Extracted from [`memory-core-demo`](https://github.com/tmzt/memory-core-demo) via `git filter-repo`, preserving the ~30 commits that touched these paths. Consumed there as a `git clone --shared repos/chitin-model-api deps/model_api` working tree — same convention as `deps/litert-rs`.

Not standalone-buildable on its own yet: `model_api_server` path-deps into `common` + `thinker_impl` from the parent workspace. This repo is the source of truth for the crate history + a hand-off point for pulling the model-api surface into other workspaces; a fully-decoupled build target is future work.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
