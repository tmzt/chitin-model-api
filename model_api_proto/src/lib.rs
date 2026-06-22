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
    /// Clean shutdown — server drains its queue, then exits.
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

/// Wire-shape of the input payload. Today only `Text` is meaningful
/// on the llama.cpp path; the audio variants are stubs reserved for
/// future routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceInput {
    Text(String),
    Pcm { samples: Vec<f32>, sample_rate: u32 },
    Mel { data: Vec<f32>, frames: u32 },
}

/// Per-request knob bag — future structured-output flags land here so
/// the outer `InferenceRequest` schema stays stable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Skip prefilling `<think>\n` at the start of the assistant turn.
    pub disable_think_prefix: bool,
    // Future: schema-pinned output, sampler overrides, etc.
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
/// `InferenceRequest::stream` is true).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Newly decoded text since the previous chunk.
    pub delta: String,
}

/// Real-time progress milestone (when `InferenceRequest::progress` is
/// true). The exact set is open — clients should treat unknown
/// variants as informational.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProgressEvent {
    Queued,
    Dispatched,
    ToolCallStarted { name: String },
    ToolCallCompleted { name: String },
    GenerationStarted,
    GenerationDone { tokens_generated: u32 },
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
    /// Tool-response payloads that the dispatcher injected during this
    /// generation, in order.
    pub injections: Vec<String>,
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
pub const PROTOCOL_VERSION: u32 = 1;
