//! Wire protocol for the model_api Unix-domain-socket server.
//!
//! The model_api server is a separate process that owns the llama.cpp
//! Session + the smart KV cache. Clients connect over UDS and exchange
//! the structs in this crate, framed length-prefixed with `bincode`.
//!
//! ## Why mirror types instead of re-exporting `common::handles`?
//!
//! The in-process `ThinkerRequest` / `ThinkerResponse` types carry
//! Rust runtime values (channel senders, Arc'd dispatchers) that don't
//! serialize and don't belong on a wire format. They also live inside
//! the `common` crate, which transitively pulls in much of the rest of
//! the workspace — bad shape for a future Python client (PyO3 /
//! PyOxidizer wrapping this proto crate stays clean only when this
//! crate's dep tree is tiny). The mirror types here are the wire
//! contract; the boundary code in `model_api_client` /
//! `model_api_server` converts to/from the in-process types.
//!
//! ## Wire framing
//!
//! Each message is `[u32 LE len][bincode payload]`. Length-prefixed so
//! both sides can stream multiple messages over the same socket
//! without scanning. See `model_api_client::framed` and
//! `model_api_server::framed` for the codec.

use serde::{Deserialize, Serialize};

// ── Top-level envelopes ─────────────────────────────────────────────

/// Anything a client sends to the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientMessage {
    /// First message on every connection. Server replies with
    /// [`ServerMessage::Hello`] reporting the model it has loaded.
    Hello { protocol_version: u32 },
    /// Submit an inference request. Server queues it, streams progress
    /// + chunks back via `ServerMessage::*`, finishes with
    /// `InferenceComplete`.
    Inference(InferenceRequest),
    /// Best-effort cancel of the current in-flight inference.
    Cancel,
    /// Close this connection cleanly. Server replies with
    /// [`ServerMessage::Goodbye`] then closes the connection. Does
    /// NOT shut down the server process — the slot's reference
    /// count drops by one; other clients continue unaffected.
    /// Functionally equivalent to dropping the socket; the wire
    /// frame just lets the server log a polite hangup instead of
    /// an abrupt EOF.
    Shutdown,
}

/// Anything the server sends to a client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerMessage {
    /// Reply to `Hello`. Carries server metadata.
    Hello {
        protocol_version: u32,
        model_name: String,
        /// Best-effort GPU memory in MB (`None` for CPU-only or unknown).
        gpu_memory_mb: Option<u32>,
    },
    /// Real-time progress milestone for the most recent
    /// [`InferenceRequest`]. One or many before `InferenceComplete`.
    Progress(ProgressEvent),
    /// Incremental decoded text (or other typed chunk). Streams while
    /// the inference is still running.
    Chunk(StreamChunk),
    /// Final inference result. Always last for a given request.
    InferenceComplete(InferenceResponse),
    /// Inference failed before completing.
    InferenceError { message: String },
    /// Server is shutting down (response to `Shutdown` or unsolicited
    /// on a SIGTERM).
    Goodbye,
}

// ── Inference types ─────────────────────────────────────────────────

/// Request body for a single inference. Mirrors the subset of
/// `common::handles::ThinkerRequest` that crosses the wire — channel
/// senders + `Arc` dispatchers stay on the client side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub role: GpuRole,
    pub input: InferenceInput,
    pub max_tokens: u32,
    pub session: SessionMode,
    pub inference_config: InferenceConfig,
    /// Tool definitions the model can call this turn. The server
    /// hands these to the loaded model's `ToolFormat` adapter so the
    /// system prompt is rendered model-natively (Gemma 4 Jinja,
    /// ChatML, XML, etc.) — clients pass abstract tool defs and let
    /// the server pick the wire format. Empty = no tools available
    /// this turn.
    #[serde(default)]
    pub tools: Vec<ToolDef>,
    /// Results from tool calls the client executed in response to a
    /// prior turn's `tool_calls`. The server formats each via the
    /// loaded model's `ToolFormat::format_response` and folds them
    /// into the next user turn before generation. Empty = no
    /// pending tool results.
    #[serde(default)]
    pub tool_results: Vec<ToolResult>,
    /// Merkle root of the conversation the caller has committed. Used
    /// by the server to decide whether to re-prefill or reuse the
    /// on-disk KV cache. 0 = "caller didn't supply one".
    pub cache_hash: u64,
    /// When true, ask the server to stream `ServerMessage::Chunk`s
    /// as text becomes available. When false, the client only gets
    /// the final `InferenceComplete`.
    pub stream: bool,
    /// When true, ask for `ServerMessage::Progress` events at
    /// meaningful inflection points (queued, dispatched, tool fired,
    /// generation complete).
    pub progress: bool,
}

/// Wire-shape of the input payload.
///
/// - `Text(s)` is a single, untemplated user message — the slot may
///   prepend its own chat template (LlamaSlot path) or wrap it as a
///   single User turn (LiteRtLmSlot path). Callers MUST NOT
///   pre-render a chat template into `Text`; use `Turns` for
///   structured multi-turn input.
/// - `Turns(t)` is a model-agnostic turn list. The receiving slot
///   either renders it through the model's `ChatFormat` (LlamaSlot)
///   or feeds turns sequentially into LiteRT-LM's `Conversation`
///   (LiteRtLmSlot).
/// - `Pcm` / `Mel` are stubs reserved for future audio routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceInput {
    Text(String),
    Turns(Vec<Turn>),
    Pcm { samples: Vec<f32>, sample_rate: u32 },
    Mel { data: Vec<f32>, frames: u32 },
}

/// Role of a single conversation turn. Mirrors the OpenAI-style
/// chat-completion roles. `Tool` carries a tool result with the
/// matching `tool_call_id` set on the [`Turn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One conversation turn. The slot's `ChatFormat` is responsible for
/// rendering the role + content into the model's template; clients
/// don't need to know what tokens the model uses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub role: Role,
    pub content: String,
    /// Set when `role == Tool` so the slot can correlate the result
    /// back to the assistant turn that issued the matching call.
    /// Otherwise `None`. (No `skip_serializing_if` — bincode is
    /// positional, every field must be present on every wire frame
    /// or the next field reads into the wrong bytes.)
    pub tool_call_id: Option<String>,
}

impl Turn {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into(), tool_call_id: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into(), tool_call_id: None }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into(), tool_call_id: None }
    }
    pub fn tool(content: impl Into<String>, call_id: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_call_id: Some(call_id.into()),
        }
    }
}

/// Per-request knob bag. Mirrors `common::handles::InferenceConfig`'s
/// sampler + structure fields; new flags land here so the outer
/// `InferenceRequest` schema stays stable. All defaults match the
/// in-process side so an empty `InferenceConfig` reproduces the
/// server's baked-in behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Skip prefilling `<think>\n` at the start of the assistant turn.
    /// Useful for plain chat-completion APIs that don't want a
    /// chain-of-thought block in the response.
    #[serde(default)]
    pub disable_think_prefix: bool,
    /// Output-shape constraint. `None` = free-form text.
    #[serde(default)]
    pub json_mode: JsonMode,
    /// Softmax temperature. Lower = more deterministic. `None` =
    /// server default (~0.7).
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus cumulative probability. `None` = server default (~0.8).
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Repetition penalty. `None` = 1.0 (off).
    #[serde(default)]
    pub rep_penalty: Option<f32>,
    /// Presence penalty. `None` = server default (~1.5).
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Per-request token cap override. `None` =
    /// `InferenceRequest::max_tokens` is the only cap.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// System-prompt override for this turn. `None` = use the
    /// session/role baked-in system prompt.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// How the server should handle tool calls the model emits
    /// during generation. See [`ToolMode`].
    #[serde(default)]
    pub tool_mode: ToolMode,
    /// Per-N-chars accumulator for streamed [`StreamChunk`]s. The
    /// decoder buffers tokens until this many UTF-8 chars have
    /// landed in the visible-text stream, then flushes a chunk.
    /// `None` = server default (currently 16). Smaller = smoother
    /// UI + more wire frames; larger = quieter wire. Only meaningful
    /// when `InferenceRequest::stream` is true.
    #[serde(default)]
    pub stream_chunk_chars: Option<u32>,
    /// When true, also emit `StreamChunk { phase: Some("thinking") }`
    /// for content inside `<think>...</think>` blocks. Default
    /// false — most UIs only want the final visible text. Has no
    /// effect when `InferenceRequest::stream` is false.
    #[serde(default)]
    pub stream_thinking: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            disable_think_prefix: false,
            json_mode: JsonMode::default(),
            temperature: None,
            top_p: None,
            rep_penalty: None,
            presence_penalty: None,
            max_tokens: None,
            system_prompt: None,
            tool_mode: ToolMode::default(),
            stream_chunk_chars: None,
            stream_thinking: false,
        }
    }
}

/// Output-shape constraint. Mirrors `common::handles::JsonMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum JsonMode {
    /// No constraint. Free-form text.
    #[default]
    None,
    /// Force a single `<tool_call>{...}</tool_call>` reply, nothing
    /// else. No `<think>` prefix.
    ToolOnly,
    /// Like `ToolOnly` but prefills `<think>\n` first so the model
    /// reasons before dispatching.
    ThinkingWithTools,
    /// Force a single JSON object, no surrounding markers or prose.
    /// No key-name constraint.
    AnyJSON,
    /// Notes-classifier-specific schema: `{"project":..., "topics":...,
    /// "summary":...}`.
    NotesClassifierJSON,
}

/// Where tool calls get dispatched. See [`InferenceConfig::tool_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ToolMode {
    /// Auto-pick: `Client` when [`InferenceRequest::tools`] is
    /// non-empty, `Server` otherwise. Default.
    #[default]
    Auto,
    /// Server dispatches tool calls mid-generation via its built-in
    /// catalog (memory, project graph, etc.). The client never sees
    /// the calls; the model's continuation reflects the tool
    /// results. Matches today's `EpiphanyDispatcher` behaviour.
    Server,
    /// Server buffers tool calls and returns them in
    /// [`InferenceResponse::tool_calls`]. Generation stops at the
    /// first call. Client executes the tool externally and replies
    /// with [`InferenceRequest::tool_results`] on the next turn.
    /// Matches OpenAI's function-calling loop — what agent clients
    /// (pi-ai, etc.) expect.
    Client,
}

// ── Tool wire types ─────────────────────────────────────────────────

/// Tool definition the model can call. Mirrors
/// `common::types::ExternalToolDef` field-for-field so the server
/// can hand it straight to `ToolFormat::system_prompt_fragment`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    /// Stable identifier the model emits when it wants to call this
    /// tool.
    pub name: String,
    /// Free-form description rendered into the system prompt's tool
    /// catalog. Keep short; the model reads it every turn.
    pub description: String,
    /// Ordered parameter list. Order matters — some formats render
    /// the parameters positionally.
    pub parameters: Vec<ToolParam>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParam {
    pub name: String,
    /// Loose type string: `"string"`, `"integer"`, `"boolean"`,
    /// `"number"`, `"array"`, `"object"`. Matches what the
    /// per-model `ToolFormat` rendering paths consume.
    #[serde(rename = "type")]
    pub param_type: String,
    pub required: bool,
    pub description: String,
}

/// A tool the model wants the client to execute. Returned in
/// [`InferenceResponse::tool_calls`] when
/// [`ToolMode::Client`] is in effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Server-assigned identifier. Echo back as
    /// [`ToolResult::call_id`] on the next request so the server can
    /// pair results to calls.
    pub id: String,
    pub name: String,
    /// JSON-encoded argument object. Parse with `JSON.parse(...)`
    /// in JS or `serde_json::from_str` in Rust before invoking the
    /// tool.
    pub arguments_json: String,
}

/// Result of executing a tool call. Carried in
/// [`InferenceRequest::tool_results`] on the next turn after the
/// model emitted [`ToolCall`]s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Matches [`ToolCall::id`] on the corresponding call.
    pub call_id: String,
    /// String the server wraps via
    /// `ToolFormat::format_response(output)` and folds back into
    /// the model's context. Free-form — the tool decides whether
    /// to return raw text, JSON, an error message, etc.
    pub output: String,
}

/// How the server should treat the session's KV cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionMode {
    /// One-shot — no session lookup, no save.
    Stateless,
    /// Use the named session: load on entry if it exists on disk,
    /// save on exit.
    Persistent { session_id: String },
}

/// Which model role this request is for. Server checks against the
/// loaded model and rejects mismatches with `InferenceError`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuRole {
    Fast = 1,
    Deep = 2,
    Asr = 4,
    Omni = 8,
}

/// Streamed text chunk emitted during generation (when
/// `InferenceRequest::stream` is true). Mirrors
/// `common::handles::StreamChunk`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Newly decoded text since the previous chunk.
    pub delta_text: String,
    /// Stop signal: `None` while generating, `Some` on the last chunk.
    pub finish_reason: Option<String>,
    /// Which sub-stream this delta belongs to. `None` and
    /// `Some("text")` are equivalent — visible answer text the
    /// final `InferenceResponse.text` will also contain.
    /// `Some("thinking")` is emitted only when the request set
    /// `inference_config.stream_thinking = true` and the model is
    /// inside a `<think>...</think>` block; UIs typically show
    /// these in a separate "reasoning" pane.
    #[serde(default)]
    pub phase: Option<String>,
}

/// Real-time progress milestone (when `InferenceRequest::progress` is
/// true). Mirrors `common::handles::ProgressEvent` — string-shaped
/// for forward compatibility: clients that don't recognize a `phase`
/// value should treat it as informational rather than error out.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressEvent {
    /// One of `queued`, `dispatched`, `tool_start`, `tool_end`,
    /// `gen_start`, `gen_done`, or any future server-defined value.
    pub phase: String,
    /// Tool name for tool-related phases.
    pub tool: Option<String>,
    /// Free-form descriptive detail (e.g. token count).
    pub detail: Option<String>,
}

/// Final inference result. Mirrors `common::handles::ThinkerResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Cleaned answer text (think blocks already stripped).
    pub text: String,
    /// Session ID if a session was created or continued.
    pub session_id: Option<String>,
    /// Raw decoded model output, including `<think>...</think>` and
    /// any injected `<tool_response>...</tool_response>` blocks.
    pub raw_text: Option<String>,
    /// Tool-response payloads that the *server-side* dispatcher
    /// injected during this generation, in order. Always empty when
    /// [`ToolMode::Client`] is in effect — those calls land in
    /// `tool_calls` instead.
    pub injections: Vec<String>,
    /// Tool calls the model emitted that the client needs to
    /// execute. Populated when [`ToolMode::Client`] is in effect
    /// (or `Auto` with tools present). Empty otherwise.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Set when the inference loop terminated because the dispatcher
    /// signalled a session hand-off.
    pub replacement: Option<SessionReplacement>,
}

/// Hand-off marker — see `common::handles::SessionReplacement` for
/// the in-process equivalent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionReplacement {
    pub template_session_id: String,
    pub input: String,
}

// ── Wire version ────────────────────────────────────────────────────

/// Bump when adding/removing variants or fields. Client + server
/// negotiate via [`ClientMessage::Hello`] / [`ServerMessage::Hello`].
///
/// Version history:
///   1 — initial. Hello/Inference/Cancel/Shutdown envelopes;
///       single `disable_think_prefix` knob in InferenceConfig.
///   2 — adds tool definitions on requests (`tools`,
///       `tool_results`), tool calls on responses (`tool_calls`),
///       expanded InferenceConfig (temperature/top_p/rep_penalty/
///       presence_penalty/max_tokens/json_mode/system_prompt/
///       tool_mode), and JsonMode + ToolMode enums.
///   3 — adds `InferenceInput::Turns(Vec<Turn>)` and the `Role` /
///       `Turn` types so clients can ship model-agnostic turn lists
///       instead of pre-rendered chat templates. `Text` is
///       redefined as "single untemplated user message" (clients
///       MUST NOT pre-template). Slots are now the only layer that
///       knows the model's chat-template tokens.
pub const PROTOCOL_VERSION: u32 = 3;
