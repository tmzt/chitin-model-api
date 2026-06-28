//! `SlotHandle` backed by the `litertlm` crate (Google LiteRT-LM
//! Rust bindings to the C engine API). Used on the Pixel demo to
//! run Gemma-4 (E2B / E4B) on the PowerVR GPU via OpenCL — the
//! path that llama.cpp's Vulkan + OpenCL backends can't handle.
//!
//! Engine + Session live for the server's lifetime; we create one
//! Session per request from the shared Engine (Session is the
//! stateful conversation thing — for stateless completions we
//! discard it after generate()). The underlying C API is sync, so
//! both `run` and `run_stream` use `smol::unblock` to park the
//! actual generate call on the blocking pool — keeps the async
//! executor free to drain the wire.
//!
//! Sessions don't share KV cache today; each request re-tokenizes
//! and re-prefills the prompt. Multi-turn continuity (the wire's
//! `SessionMode::Persistent`) would need an Engine::create_conversation
//! pool keyed by session_id — wire that when a caller actually
//! needs it.

use std::sync::Arc;

use async_trait::async_trait;
use litertlm::{Backend, Engine, EngineSettings, SamplerParams};
use model_api_proto::{InferenceInput, InferenceResponse, StreamChunk};

use crate::slot::{SlotHandle, SlotRequest, SlotResponse, StreamSink};

pub struct LiteRtLmSlot {
    engine: Arc<Engine>,
    model_name: String,
    /// CLI-supplied default sampler params. Per-request inference
    /// config overrides these in `sampler_for_request`.
    default_sampler: SamplerParams,
    /// Per-Conversation visual token budget passed via
    /// `LiteRtLmConversationOptionalArgs`. Multimodal Gemma-4
    /// models reject text-only sends without it. `None` -> rely on
    /// model defaults (works for pure-text models).
    visual_token_budget: Option<i32>,
}

// We use `Conversation` rather than `Session` because Session::generate
// feeds the prompt to the model raw and instruction-tuned models like
// Gemma silently produce nothing (the C API returns null). Conversation
// wraps the upstream C `litert_lm_conversation_*` API which applies
// the model's chat template (`chat_template.jinja` baked into the
// `.litertlm` file).

impl LiteRtLmSlot {
    pub fn new(
        model_path: std::path::PathBuf,
        model_name: String,
        backend: Backend,
        max_num_tokens: i32,
        visual_token_budget: Option<i32>,
    ) -> Result<Self, String> {
        let settings = EngineSettings::new(model_path)
            .backend(backend)
            .max_num_tokens(max_num_tokens);
        let engine = Engine::new(settings)
            .map_err(|e| format!("litertlm Engine::new: {e}"))?;
        // v0.13's sampler_factory only implements TYPE_UNSPECIFIED
        // (no-op) and TOP_P; TopK and Greedy return UnimplementedError.
        // SamplerParams::default() picks TopK, which would blow up,
        // so override to TopP up front. The underlying C call still
        // routes through `kLiteRtLmSamplerTypeTopP`.
        let default_sampler = SamplerParams::default().top_p(0.95);
        Ok(Self {
            engine: Arc::new(engine),
            model_name,
            default_sampler,
            visual_token_budget,
        })
    }

    fn sampler_for_request(&self, req: &SlotRequest) -> SamplerParams {
        let mut s = self.default_sampler.clone();
        if let Some(t) = req.req.inference_config.temperature {
            s = s.temperature(t);
        }
        if let Some(p) = req.req.inference_config.top_p {
            s = s.top_p(p);
        }
        s
    }

    fn extract_prompt(input: &InferenceInput) -> Result<String, String> {
        match input {
            InferenceInput::Text(t) => Ok(t.clone()),
            InferenceInput::Pcm { .. } | InferenceInput::Mel { .. } => {
                Err("audio inputs not supported on litert-lm backend".into())
            }
        }
    }
}

#[async_trait]
impl SlotHandle for LiteRtLmSlot {
    fn model_name(&self) -> &str { &self.model_name }
    fn gpu_memory_mb(&self) -> Option<u32> { None }

    async fn run(&self, req: SlotRequest) -> Result<SlotResponse, String> {
        let prompt = Self::extract_prompt(&req.req.input)?;
        let sampler = self.sampler_for_request(&req);
        let engine = self.engine.clone();
        let vtb = self.visual_token_budget;
        smol::unblock(move || {
            let mut conv = engine
                .create_conversation(sampler)
                .map_err(|e| format!("litertlm create_conversation: {e}"))?;
            if let Some(b) = vtb { conv = conv.with_visual_token_budget(b); }
            let text = conv
                .send_message(&prompt)
                .map_err(|e| format!("litertlm send_message: {e}"))?;
            Ok::<SlotResponse, String>(SlotResponse(InferenceResponse {
                text,
                session_id: None,
                raw_text: None,
                injections: Vec::new(),
                tool_calls: Vec::new(),
                replacement: None,
            }))
        }).await
    }

    async fn run_stream(
        &self,
        req: SlotRequest,
        sink: &dyn StreamSink,
    ) -> Result<SlotResponse, String> {
        let prompt = Self::extract_prompt(&req.req.input)?;
        let sampler = self.sampler_for_request(&req);
        let engine = self.engine.clone();

        // bounded(64) backpressures the C callback when the wire
        // can't keep up with token emission — same sizing as
        // LlamaSlot's stream/progress channels.
        let (tx, rx) = async_channel::bounded::<String>(64);

        let vtb = self.visual_token_budget;
        let gen_task = smol::unblock(move || {
            let mut conv = engine
                .create_conversation(sampler)
                .map_err(|e| format!("litertlm create_conversation: {e}"))?;
            if let Some(b) = vtb { conv = conv.with_visual_token_budget(b); }
            let mut accumulated = String::new();
            conv
                .send_message_stream(&prompt, |chunk: &str| {
                    accumulated.push_str(chunk);
                    // The callback can't signal cancellation (Conversation's
                    // streaming API takes FnMut(&str) -> ()). We park on the
                    // bounded channel so the C side blocks if the drain
                    // can't keep up — that's our backpressure. send() can
                    // only error when the rx is dropped, which means the
                    // server connection is gone; nothing to do at that point.
                    let _ = smol::block_on(tx.send(chunk.to_string()));
                })
                .map_err(|e| format!("litertlm send_message_stream: {e}"))?;
            // Drop the tx so the drain side sees EOF on the channel.
            drop(tx);
            Ok::<String, String>(accumulated)
        });

        let drain = async {
            while let Ok(chunk) = rx.recv().await {
                sink.on_chunk(StreamChunk {
                    delta_text: chunk,
                    finish_reason: None,
                    phase: Some("text".into()),
                });
            }
        };

        let (gen_result, ()) = futures_lite::future::zip(gen_task, drain).await;
        let text = gen_result?;

        // Final marker chunk so wire consumers know we're done
        // without having to wait for InferenceComplete.
        sink.on_chunk(StreamChunk {
            delta_text: String::new(),
            finish_reason: Some("stop".into()),
            phase: Some("text".into()),
        });

        Ok(SlotResponse(InferenceResponse {
            text,
            session_id: None,
            raw_text: None,
            injections: Vec::new(),
            tool_calls: Vec::new(),
            replacement: None,
        }))
    }
}
