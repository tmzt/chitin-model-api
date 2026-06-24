//! Slot abstraction — what the connection handler depends on
//! instead of pulling llama_engine in directly. Keeps the handler
//! testable with a stub.
//!
//! Production: [`crate::llama_slot::LlamaSlotHandle`] (under the
//! `llama-cpp` feature) wraps `thinker_impl::spawn_resource`.
//! Tests: [`StubSlot`] just echoes the request as a response.

use async_trait::async_trait;
use model_api_proto::{InferenceRequest, InferenceResponse};

/// What the proto's `InferenceRequest` looks like inside the server.
/// Today this is a passthrough; the indirection lets us add internal
/// fields (request id, peer label, deadline) without churning the
/// wire type.
#[derive(Debug)]
pub struct SlotRequest {
    pub req: InferenceRequest,
}

impl SlotRequest {
    pub fn from_proto(req: InferenceRequest) -> Self {
        Self { req }
    }
}

/// Slot's reply, ready to wrap in `ServerMessage::InferenceComplete`.
#[derive(Debug)]
pub struct SlotResponse(pub InferenceResponse);

/// The thing the connection handler talks to. One instance per
/// running server; the server holds it in an `Arc` and clones per
/// connection so all connections feed into the same slot.
#[async_trait]
pub trait SlotHandle: Send + Sync {
    /// Human-readable model name (e.g. `"gemma-4-26b"`, `"stub"`).
    /// Returned to the client in the `Hello` reply.
    fn model_name(&self) -> &str;
    /// Best-effort GPU memory in MB. `None` for CPU-only or unknown.
    fn gpu_memory_mb(&self) -> Option<u32>;
    /// Run one inference. Returns either a final response or a
    /// human-readable error string. Slot is single-threaded today —
    /// callers serialize naturally.
    async fn run(&self, req: SlotRequest) -> Result<SlotResponse, String>;
}

// ── Stub for tests ──────────────────────────────────────────────────

/// A slot that echoes the prompt back as the response text. Used by
/// the server's integration tests so they can exercise the full
/// connection lifecycle without loading a real GGUF model.
pub struct StubSlot {
    pub name: String,
}

impl StubSlot {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait]
impl SlotHandle for StubSlot {
    fn model_name(&self) -> &str { &self.name }
    fn gpu_memory_mb(&self) -> Option<u32> { None }
    async fn run(&self, req: SlotRequest) -> Result<SlotResponse, String> {
        // Pull the prompt text out and echo it. Anything other than
        // a Text input gets a marker so test failures are obvious.
        let prompt = match req.req.input {
            model_api_proto::InferenceInput::Text(t) => t,
            model_api_proto::InferenceInput::Pcm { .. } => "[stub: pcm]".into(),
            model_api_proto::InferenceInput::Mel { .. } => "[stub: mel]".into(),
        };
        let session_id = match req.req.session {
            model_api_proto::SessionMode::Stateless => None,
            model_api_proto::SessionMode::Persistent { session_id } => Some(session_id),
        };
        // Tool-aware echo: if the caller passed a tool def, emit a
        // canned ToolCall for the first tool so tests can exercise
        // the wire-side tool-call round-trip without a real model.
        // The id is deterministic (just `tc-0`) — tests asserting on
        // call ids stay stable across runs.
        let tool_calls = if !req.req.tools.is_empty() {
            let first = &req.req.tools[0];
            vec![model_api_proto::ToolCall {
                id: "tc-0".into(),
                name: first.name.clone(),
                arguments_json: "{}".into(),
            }]
        } else {
            Vec::new()
        };
        Ok(SlotResponse(InferenceResponse {
            text: format!("echo: {prompt}"),
            session_id,
            raw_text: Some(format!("<raw>echo: {prompt}</raw>")),
            injections: Vec::new(),
            tool_calls,
            replacement: None,
        }))
    }
}
