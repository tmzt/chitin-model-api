//! Production [`SlotHandle`] backed by `thinker_impl::spawn_resource`.
//!
//! Owns the in-process llama slot (single slot, sized at server
//! startup) and bridges the wire types in [`crate::slot::SlotRequest`]
//! /[`SlotResponse`] to `common::handles::ThinkerRequest` /
//! `ThinkerResponse`. The translation mirrors what
//! `thinker_impl::model_api_remote` does on the client side, just
//! in reverse.
//!
//! Cancellation, streaming chunks, and progress events are noted
//! TODOs — the server delivers a single InferenceComplete per
//! request today; in-band cancel + chunk streaming need plumbing
//! into thinker_impl's slot loop and land in a follow-up commit.

use std::sync::Arc;

use async_trait::async_trait;
use common::handles::{
    GpuRole as CommonRole, InferenceConfig as CommonInferenceConfig,
    InferenceInput as CommonInput, KvCacheMode, SessionMode as CommonSession,
    ThinkerRequest, ThinkerResource,
};
use common::tool_format::Gemma4ToolFormat;
use model_api_proto::{
    GpuRole, InferenceInput, InferenceResponse, SessionMode, SessionReplacement,
};
use thinker_impl::ThinkerConfig;

use crate::slot::{SlotHandle, SlotRequest, SlotResponse};

/// SlotHandle backed by `thinker_impl::spawn_resource`.
///
/// Holds the `ThinkerResource` so its model_name / gpu_memory_mb
/// surface verbatim; takes a clone of the request sender on every
/// `run()`. Sender is `Clone` (it's an `async_channel::Sender`
/// underneath), so this is cheap.
pub struct LlamaSlot {
    resource: Box<dyn ThinkerResource>,
}

impl LlamaSlot {
    /// Build a sensible default `ThinkerConfig` from the binary's
    /// `--model` / `--model-dir` CLI args and hand off to
    /// `thinker_impl::spawn_resource`. Fails synchronously if the
    /// model can't be loaded (the slot panics on first decode
    /// instead of at spawn today, but that's a thinker_impl-side
    /// concern — a future cleanup should surface model-load failures
    /// up here so the server doesn't accept connections it can't
    /// honour).
    pub fn spawn(
        model_dir: String,
        gguf_path: Option<String>,
        model_name: String,
        max_tokens: usize,
        max_seq_len: u32,
    ) -> Result<Self, String> {
        let config = ThinkerConfig {
            model_dir,
            max_tokens,
            max_seq_len,
            json_mode: false,
            // Static = prefix-cache only, no per-session persistence.
            // Server callers that want session-keyed KV pass a non-
            // Stateless wire SessionMode; the server then routes via
            // the request's SessionMode without changing the
            // role-level KV config. Disk-backed sessions need extra
            // wiring (cache_dir + system_prompt) that we'll add when
            // a caller actually needs them.
            kv_cache: KvCacheMode::Static(String::new()),
            model_name,
            gguf_path,
            mtp_head_path: None,
            mtp_draft_n: 0,
            // Gemma 4 is the current production target. Different
            // models can override later (CLI arg or per-request).
            tool_format: Arc::new(Gemma4ToolFormat),
            // Empty toolsets — the dispatcher returns "unknown tool"
            // for every call. Real tools come from a future commit
            // that wires them through the wire protocol.
            tool_sets: Arc::new(Vec::new()),
        };
        let resource = thinker_impl::spawn_resource(config);
        Ok(Self { resource })
    }
}

#[async_trait]
impl SlotHandle for LlamaSlot {
    fn model_name(&self) -> &str { self.resource.model_name() }
    fn gpu_memory_mb(&self) -> Option<u32> { self.resource.gpu_memory_mb() }

    async fn run(&self, req: SlotRequest) -> Result<SlotResponse, String> {
        let SlotRequest { req } = req;

        // Channel for the slot to drop its single response into.
        // bounded(1) so a slow consumer can't queue up multiple
        // pending responses against a single request.
        let (resp_tx, resp_rx) = smol::channel::bounded::<common::handles::ThinkerResponse>(1);

        let thinker_req = ThinkerRequest {
            role: to_common_role(req.role),
            input: to_common_input(req.input),
            max_tokens: req.max_tokens as usize,
            response_tx: resp_tx,
            session: to_common_session(req.session),
            // Streaming + progress sinks not wired yet — server today
            // sends only the final InferenceComplete. The wire
            // request still carries `stream` / `progress` bits so
            // the slot can opt to skip extra work; we just ignore
            // them here until the matching server-side fan-out lands.
            stream_tx: None,
            progress_tx: None,
            prefix_cache: None,
            disable_think_prefix: req.inference_config.disable_think_prefix,
            inference_config: CommonInferenceConfig::default(),
            cache_hash: req.cache_hash,
        };

        let tx = self.resource.request_sender();
        if let Err(e) = tx.send(thinker_req).await {
            return Err(format!("slot channel closed: {e}"));
        }

        let common_resp = resp_rx.recv().await
            .map_err(|e| format!("slot response channel closed: {e}"))?;

        Ok(SlotResponse(InferenceResponse {
            text: common_resp.text,
            session_id: common_resp.session_id,
            raw_text: common_resp.raw_text,
            injections: common_resp.injections,
            replacement: common_resp.replacement.map(|r| SessionReplacement {
                template_session_id: r.template_session_id,
                input: r.input,
            }),
        }))
    }
}

// ── Translation helpers ─────────────────────────────────────────────

fn to_common_role(r: GpuRole) -> CommonRole {
    match r {
        GpuRole::Fast => CommonRole::Fast,
        GpuRole::Deep => CommonRole::Deep,
        GpuRole::Asr  => CommonRole::Asr,
        GpuRole::Omni => CommonRole::Omni,
    }
}

fn to_common_input(i: InferenceInput) -> CommonInput {
    match i {
        InferenceInput::Text(t) => CommonInput::Text(t),
        InferenceInput::Pcm { samples, sample_rate } => CommonInput::Pcm { samples, sample_rate },
        InferenceInput::Mel { data, frames } => CommonInput::Mel { data, frames },
    }
}

fn to_common_session(s: SessionMode) -> CommonSession {
    match s {
        SessionMode::Stateless => CommonSession::NoSession,
        SessionMode::Persistent { session_id } => {
            if session_id.is_empty() {
                CommonSession::NewSession
            } else {
                CommonSession::DiskBasedSession(session_id)
            }
        }
    }
}
