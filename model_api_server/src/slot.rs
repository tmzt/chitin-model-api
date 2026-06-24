//! Slot abstraction — what the connection handler depends on
//! instead of pulling llama_engine in directly. Keeps the handler
//! testable with a stub.
//!
//! Production: [`crate::llama_slot::LlamaSlotHandle`] (under the
//! `llama-cpp` feature) wraps `thinker_impl::spawn_resource`.
//! Tests: [`StubSlot`] just echoes the request as a response.

use async_trait::async_trait;
use model_api_proto::{InferenceRequest, InferenceResponse, ProgressEvent, StreamChunk};

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

/// Sink the slot pushes streaming events into during a generation.
/// The server's connection handler implements this with a closure
/// that forwards each event to the wire as `ServerMessage::Chunk` /
/// `ServerMessage::Progress` frames.
///
/// Trait (not closure) so impls can hold state — e.g. the wire
/// adapter holds the writer half + the connection id for logging.
/// `Send` because the slot may invoke the sink from a different
/// thread (the llama slot runs on its own thread); `Sync` for the
/// same reason if the sink is cloned per slot worker.
pub trait StreamSink: Send + Sync {
    fn on_chunk(&self, chunk: StreamChunk);
    fn on_progress(&self, progress: ProgressEvent);
}

/// A `StreamSink` that just drops everything. Useful default for
/// slots whose backend doesn't actually stream — they can still
/// satisfy the trait signature without forcing every consumer to
/// pass a real sink.
pub struct DiscardSink;
impl StreamSink for DiscardSink {
    fn on_chunk(&self, _: StreamChunk) {}
    fn on_progress(&self, _: ProgressEvent) {}
}

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
    /// Streaming variant — same semantics as `run`, but the slot
    /// pushes incremental chunks + progress events through `sink`
    /// before returning the final response. The connection handler
    /// calls this when the wire request set `stream` or `progress`.
    ///
    /// Default impl: invoke `run` and skip the sink — for slots
    /// whose backend doesn't surface incremental events yet (the
    /// llama backend today). Subscribing clients still get the
    /// final response; they just won't see any Chunk / Progress
    /// frames in between.
    async fn run_stream(
        &self,
        req: SlotRequest,
        _sink: &dyn StreamSink,
    ) -> Result<SlotResponse, String> {
        self.run(req).await
    }
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
        Ok(SlotResponse(synth_echo_response(&req)))
    }
    async fn run_stream(
        &self,
        req: SlotRequest,
        sink: &dyn StreamSink,
    ) -> Result<SlotResponse, String> {
        // Synthetic progress + chunk emission so the wire-side
        // streaming path can be tested end-to-end without a real
        // model. Sequence:
        //   1. Progress { phase: "queued" }
        //   2. Progress { phase: "gen_start" }
        //   3. Chunk × N — split the echoed prompt into ~3 chunks
        //   4. Progress { phase: "gen_done", detail: <token_count> }
        //   5. Final response (via the non-streaming `run`).
        sink.on_progress(ProgressEvent {
            phase: "queued".into(), tool: None, detail: None,
        });
        sink.on_progress(ProgressEvent {
            phase: "gen_start".into(), tool: None, detail: None,
        });
        let resp = synth_echo_response(&req);
        // Cut the final text into 3 roughly equal pieces and emit
        // them as chunks so consumers see >1 delta.
        let text = resp.text.clone();
        let n = text.chars().count().max(1);
        let chunk_size = (n / 3).max(1);
        let mut buf = String::new();
        for (i, ch) in text.chars().enumerate() {
            buf.push(ch);
            if (i + 1) % chunk_size == 0 || i + 1 == n {
                sink.on_chunk(StreamChunk {
                    delta_text: std::mem::take(&mut buf),
                    finish_reason: if i + 1 == n { Some("stop".into()) } else { None },
                });
            }
        }
        sink.on_progress(ProgressEvent {
            phase: "gen_done".into(), tool: None,
            detail: Some(format!("{n} chars")),
        });
        Ok(SlotResponse(resp))
    }
}

/// Lifted out so both `run` and `run_stream` produce the same final
/// response. Echo + optional tool-call emission, deterministic.
fn synth_echo_response(req: &SlotRequest) -> InferenceResponse {
    // Pull the prompt text out and echo it. Anything other than
    // a Text input gets a marker so test failures are obvious.
    let prompt = match &req.req.input {
        model_api_proto::InferenceInput::Text(t) => t.clone(),
        model_api_proto::InferenceInput::Pcm { .. } => "[stub: pcm]".into(),
        model_api_proto::InferenceInput::Mel { .. } => "[stub: mel]".into(),
    };
    let session_id = match &req.req.session {
        model_api_proto::SessionMode::Stateless => None,
        model_api_proto::SessionMode::Persistent { session_id } => Some(session_id.clone()),
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
    InferenceResponse {
        text: format!("echo: {prompt}"),
        session_id,
        raw_text: Some(format!("<raw>echo: {prompt}</raw>")),
        injections: Vec::new(),
        tool_calls,
        replacement: None,
    }
}
