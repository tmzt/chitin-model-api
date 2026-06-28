//! Production [`SlotHandle`] backed by `thinker_impl::spawn_resource`.
//!
//! Owns the in-process llama slot (single slot, sized at server
//! startup) and bridges the wire types in [`crate::slot::SlotRequest`]
//! /[`SlotResponse`] to `common::handles::ThinkerRequest` /
//! `ThinkerResponse`. The translation mirrors what
//! `thinker_impl::model_api_remote` does on the client side, just in
//! reverse.
//!
//! Tool calls are handled per-request via [`proto::ToolMode`]:
//!  - `Server` — leave `raw_tool_call` off, in-band dispatcher
//!     executes calls server-side, results are spliced into the
//!     model's continuation. Response carries `injections` but not
//!     `tool_calls`.
//!  - `Client` — set `raw_tool_call` so the model's tool-call
//!     markers stay in the output. After generation, scan the text
//!     for `<tool_open>...<tool_close>` pairs using the loaded
//!     model's [`ToolFormat`], extract each into a `ToolCall`, and
//!     ship them in the response. Client (e.g. pi-ai) executes the
//!     tool externally and sends the result back via
//!     [`InferenceRequest::tool_results`] on the next turn.
//!  - `Auto` — `Client` when the request carries tools, else
//!     `Server`. Matches the OpenAI-style agent-loop expectation.
//!
//! Cancellation, streaming chunks, and progress events are still
//! TODOs — the server delivers a single InferenceComplete per
//! request today.

use std::sync::Arc;

use async_trait::async_trait;
use common::handles::{
    GpuRole as CommonRole, InferenceConfig as CommonInferenceConfig,
    InferenceInput as CommonInput, JsonMode as CommonJsonMode, KvCacheMode,
    SessionMode as CommonSession, ThinkerRequest, ThinkerResource,
};
use common::chat_format::{ChatFormat, Gemma4Format};
use common::tool_format::{Gemma4ToolFormat, ToolFormat};
use common::types::{ExternalToolDef, ExternalToolParam};
use model_api_proto::{
    GpuRole, InferenceInput, InferenceResponse, JsonMode, SessionMode,
    SessionReplacement, ToolDef, ToolMode,
};
use thinker_impl::ThinkerConfig;

use crate::slot::{SlotHandle, SlotRequest, SlotResponse, StreamSink};
use crate::{tool_text, turn_render};

/// SlotHandle backed by `thinker_impl::spawn_resource`.
///
/// Holds the `ThinkerResource` so its model_name / gpu_memory_mb
/// surface verbatim. Also holds an `Arc<dyn ToolFormat>` matching
/// the loaded model so the client-mode tool-call path can parse
/// the model's raw output without round-tripping through
/// thinker_impl.
pub struct LlamaSlot {
    resource: Box<dyn ThinkerResource>,
    /// Active tool format for the loaded model. Used by
    /// [`tool_text::extract_tool_calls`] to parse markers out of
    /// `raw_text` when the wire request asked for client-side tool
    /// dispatch, and by [`tool_text::tool_results_to_turns`] to
    /// wrap incoming tool results before they're spliced into the
    /// turn list.
    tool_format: Arc<dyn ToolFormat>,
    /// Active chat format — the per-role marker template
    /// (e.g. Gemma 4's `<|turn>{role}\n…<turn|>`). Used by
    /// [`turn_render::render_turns`] to convert
    /// `InferenceInput::Turns` into the raw-prompt string
    /// thinker_impl expects.
    chat_format: Arc<dyn ChatFormat>,
}

impl LlamaSlot {
    /// Build a sensible default `ThinkerConfig` from the binary's
    /// `--model` / `--model-dir` CLI args and hand off to
    /// `thinker_impl::spawn_resource`. Fails synchronously if the
    /// model can't be loaded.
    pub fn spawn(
        model_dir: String,
        gguf_path: Option<String>,
        model_name: String,
        max_tokens: usize,
        max_seq_len: u32,
    ) -> Result<Self, String> {
        // Gemma 4 is the current production target; the per-model
        // ToolFormat + ChatFormat are the abstraction points for
        // other models. To support a different model, add a CLI flag
        // that picks the right pair here (and adjust the channels
        // driver's marker detection — same Arc<dyn ToolFormat>
        // value).
        let tool_format: Arc<dyn ToolFormat> = Arc::new(Gemma4ToolFormat);
        let chat_format: Arc<dyn ChatFormat> = Arc::new(Gemma4Format);

        let config = ThinkerConfig {
            model_dir,
            max_tokens,
            max_seq_len,
            json_mode: false,
            // Static = prefix-cache only, no per-session persistence.
            // Per-request session continuity is handled via
            // `SessionMode::Persistent { session_id }` on the wire,
            // which translates to `CommonSession::DiskBasedSession`
            // — disk-backed sessions need extra wiring (cache_dir +
            // system_prompt) that we'll add when a caller actually
            // needs them.
            kv_cache: KvCacheMode::Static(String::new()),
            model_name,
            gguf_path,
            mtp_head_path: None,
            mtp_draft_n: 0,
            tool_format: tool_format.clone(),
            // Empty toolsets — the in-band dispatcher returns
            // "unknown tool" for every call. When ToolMode::Server
            // is in effect and the model emits a tool call, the
            // request's per-turn `inference_config.tools` overrides
            // this baked-in (empty) catalog.
            tool_sets: Arc::new(Vec::new()),
        };
        let resource = thinker_impl::spawn_resource(config);
        Ok(Self { resource, tool_format, chat_format })
    }
}

#[async_trait]
impl SlotHandle for LlamaSlot {
    fn model_name(&self) -> &str { self.resource.model_name() }
    fn gpu_memory_mb(&self) -> Option<u32> { self.resource.gpu_memory_mb() }

    async fn run(&self, req: SlotRequest) -> Result<SlotResponse, String> {
        self.run_inner(req, None).await
    }

    async fn run_stream(
        &self,
        req: SlotRequest,
        sink: &dyn StreamSink,
    ) -> Result<SlotResponse, String> {
        self.run_inner(req, Some(sink)).await
    }
}

impl LlamaSlot {
    /// Unified implementation: when `sink` is `Some`, builds an
    /// async_channel pair, plumbs the sender into ThinkerRequest.
    /// stream_tx + progress_tx, and spawns a drain task that
    /// forwards each event to the sink. When `sink` is `None`, the
    /// channels stay `None` and the slot behaves as before.
    async fn run_inner(
        &self,
        req: SlotRequest,
        sink: Option<&dyn StreamSink>,
    ) -> Result<SlotResponse, String> {
        let SlotRequest { req } = req;

        // Decide tool-dispatch mode up front so we know whether to
        // set `raw_tool_call` on the inner ThinkerRequest and
        // whether to parse markers out of the response afterward.
        let want_client_dispatch = match req.inference_config.tool_mode {
            ToolMode::Client => true,
            ToolMode::Server => false,
            ToolMode::Auto => !req.tools.is_empty(),
        };

        // Convert wire ToolDefs → ExternalToolDefs for the
        // per-turn override that thinker_impl honours.
        let common_tools = if req.tools.is_empty() {
            None
        } else {
            Some(req.tools.iter().map(to_common_tool_def).collect::<Vec<_>>())
        };

        // Two prompt assembly paths depending on the wire input:
        // - Turns: use the model's ChatFormat to render the full
        //   turn list (tool results spliced in as Role::Tool turns
        //   first). thinker_impl's `<|turn>`-detection at
        //   llama_slot.rs:380 sees pre-wrapped markers and skips
        //   its own templating — no double-wrap.
        // - Text: legacy path. Just prepend formatted tool results
        //   to the raw user text; thinker_impl wraps the whole
        //   thing in one user turn.
        let prompt = match &req.input {
            InferenceInput::Turns(turns) => {
                let mut all = turns.clone();
                all.extend(tool_text::tool_results_to_turns(
                    &req.tool_results, &*self.tool_format));
                turn_render::render_turns(
                    &*self.chat_format,
                    &all,
                    req.inference_config.system_prompt.as_deref(),
                )
            }
            InferenceInput::Text(s) => tool_text::prepend_tool_results_to_text(
                s, &req.tool_results, &*self.tool_format),
            InferenceInput::Pcm { .. } | InferenceInput::Mel { .. } => {
                return Err("audio inputs not supported on the llama backend".into());
            }
        };

        // Inference config — wire values override the defaults
        // when present. None on the wire = "use server default"
        // (in-process InferenceConfig::default reproduces the
        // historical baked constants).
        let mut common_cfg = CommonInferenceConfig::default();
        common_cfg.disable_think_prefix = req.inference_config.disable_think_prefix;
        common_cfg.json_mode = to_common_json_mode(req.inference_config.json_mode);
        if let Some(t) = req.inference_config.temperature { common_cfg.temperature = t; }
        if let Some(p) = req.inference_config.top_p { common_cfg.top_p = p; }
        if let Some(r) = req.inference_config.rep_penalty { common_cfg.rep_penalty = r; }
        if let Some(p) = req.inference_config.presence_penalty { common_cfg.presence_penalty = p; }
        if let Some(n) = req.inference_config.max_tokens { common_cfg.max_tokens = Some(n as usize); }
        common_cfg.system_prompt = req.inference_config.system_prompt.clone();
        common_cfg.tools = common_tools;
        common_cfg.raw_tool_call = want_client_dispatch;

        // Channel for the slot to drop its single response into.
        let (resp_tx, resp_rx) = smol::channel::bounded::<common::handles::ThinkerResponse>(1);

        // Streaming wiring: only set up the chunk + progress
        // channels when a sink is supplied. The channels driver
        // skips chunk emission entirely on a None stream_tx, so
        // non-streaming requests stay zero-overhead.
        let (stream_tx, stream_rx) = if sink.is_some() {
            let (tx, rx) = async_channel::bounded::<common::handles::StreamChunk>(64);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        let (progress_tx, progress_rx) = if sink.is_some() {
            let (tx, rx) = async_channel::bounded::<common::handles::ProgressEvent>(64);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        let thinker_req = ThinkerRequest {
            role: to_common_role(req.role),
            input: CommonInput::Text(prompt),
            max_tokens: req.max_tokens as usize,
            response_tx: resp_tx,
            session: to_common_session(req.session),
            stream_tx,
            progress_tx,
            prefix_cache: None,
            disable_think_prefix: req.inference_config.disable_think_prefix,
            inference_config: common_cfg,
            cache_hash: req.cache_hash,
        };

        let tx = self.resource.request_sender();
        if let Err(e) = tx.send(thinker_req).await {
            return Err(format!("slot channel closed: {e}"));
        }

        // Streaming flow: poll the response channel while draining
        // pending chunk + progress events into the sink. Yields
        // between try_recv attempts so the slot thread and event
        // emitters can make progress.
        //
        // Non-streaming flow: a plain await on resp_rx — the
        // channels are None and nothing else competes for cycles.
        let common_resp = match (stream_rx.as_ref(), progress_rx.as_ref(), sink) {
            (Some(srx), Some(prx), Some(sink_ref)) => {
                let resp = loop {
                    drain_chunks(srx, sink_ref);
                    drain_progress(prx, sink_ref);
                    match resp_rx.try_recv() {
                        Ok(r) => break r,
                        Err(async_channel::TryRecvError::Empty) => {
                            smol::future::yield_now().await;
                        }
                        Err(async_channel::TryRecvError::Closed) => {
                            return Err("slot response channel closed".into());
                        }
                    }
                };
                // Tail drain. The channels driver emits its final
                // "stop" chunk AFTER pushing the ThinkerResponse,
                // so a client that grabs the response and stops
                // listening would miss it without this loop.
                drain_chunks(srx, sink_ref);
                drain_progress(prx, sink_ref);
                resp
            }
            _ => {
                resp_rx.recv().await
                    .map_err(|e| format!("slot response channel closed: {e}"))?
            }
        };

        // Extract tool calls from the raw model output when we're in
        // client-dispatch mode. The channels driver leaves markers
        // verbatim in `text` (raw_tool_call short-circuits the
        // marker-stripping state machine), so we scan with the
        // active model's ToolFormat.
        let tool_calls = if want_client_dispatch {
            tool_text::extract_tool_calls(&common_resp.text, &*self.tool_format)
        } else {
            Vec::new()
        };

        Ok(SlotResponse(InferenceResponse {
            text: common_resp.text,
            session_id: common_resp.session_id,
            raw_text: common_resp.raw_text,
            injections: common_resp.injections,
            tool_calls,
            replacement: common_resp.replacement.map(|r| SessionReplacement {
                template_session_id: r.template_session_id,
                input: r.input,
            }),
        }))
    }
}

// ── Streaming drain helpers ─────────────────────────────────────────

fn drain_chunks(rx: &async_channel::Receiver<common::handles::StreamChunk>, sink: &dyn StreamSink) {
    while let Ok(c) = rx.try_recv() {
        sink.on_chunk(model_api_proto::StreamChunk {
            delta_text: c.delta_text,
            finish_reason: c.finish_reason,
            phase: c.phase,
        });
    }
}

fn drain_progress(rx: &async_channel::Receiver<common::handles::ProgressEvent>, sink: &dyn StreamSink) {
    while let Ok(p) = rx.try_recv() {
        sink.on_progress(model_api_proto::ProgressEvent {
            phase: p.phase,
            tool: p.tool,
            detail: p.detail,
        });
    }
}

// Tool result folding + tool call extraction relocated to
// `crate::tool_text` so the helpers can be unit-tested without
// pulling in the whole LlamaSlot dep tree, and so a future
// non-llama-cpp slot can reuse them.

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
        // Turns -> text fallback for the dead-code helper. Real Turns
        // handling for the LlamaSlot path lives in P4 (chat-format
        // render); this branch exists only to satisfy exhaustiveness.
        InferenceInput::Turns(turns) => CommonInput::Text(
            turns.into_iter().map(|t| t.content).collect::<Vec<_>>().join("\n"),
        ),
        InferenceInput::Pcm { samples, sample_rate } => CommonInput::Pcm { samples, sample_rate },
        InferenceInput::Mel { data, frames } => CommonInput::Mel { data, frames },
    }
}

#[allow(dead_code)]
fn _hold_to_common_input(i: InferenceInput) -> CommonInput { to_common_input(i) }

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

fn to_common_json_mode(m: JsonMode) -> CommonJsonMode {
    match m {
        JsonMode::None => CommonJsonMode::None,
        JsonMode::ToolOnly => CommonJsonMode::ToolOnly,
        JsonMode::ThinkingWithTools => CommonJsonMode::ThinkingWithTools,
        JsonMode::AnyJSON => CommonJsonMode::AnyJSON,
        JsonMode::NotesClassifierJSON => CommonJsonMode::NotesClassifierJSON,
    }
}

fn to_common_tool_def(t: &ToolDef) -> ExternalToolDef {
    ExternalToolDef {
        name: t.name.clone(),
        description: t.description.clone(),
        parameters: t.parameters.iter().map(|p| ExternalToolParam {
            name: p.name.clone(),
            param_type: p.param_type.clone(),
            required: p.required,
            description: p.description.clone(),
        }).collect(),
    }
}

// Parser unit tests moved to crate::tool_text::tests alongside the
// extract_tool_calls implementation.
