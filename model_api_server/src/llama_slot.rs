//! Production [`SlotHandle`] backed by an in-tree
//! [`chitin_llama::Session`].
//!
//! Owns the loaded llama.cpp model directly. Renders prompts with
//! `gemma_utils::ChatFormat`, generates tokens through
//! `Session::generate_raw`, and (in client mode) parses tool-call
//! markers back out with `gemma_utils::ToolFormat` — no parent-
//! workspace `thinker_impl` / `common::handles` indirection.
//!
//! Tool calls are handled per-request via [`proto::ToolMode`]:
//!   - `Server` — model output is returned as text; markers stay
//!     verbatim but no `tool_calls` are surfaced. Callers that
//!     want in-line dispatch have to layer it themselves.
//!   - `Client` — after generation, the raw text is scanned with
//!     the active `ToolFormat` and extracted `ToolCall`s ride
//!     alongside `text` on the response. Client (pi-ai, chitin's
//!     agent runtime, etc.) executes the tool and returns the
//!     result on the next turn as `InferenceRequest::tool_results`.
//!   - `Auto` — `Client` when the request carries tools, else
//!     `Server`.
//!
//! Cancellation and streaming chunk / progress events are still
//! TODOs — the slot delivers a single response per request today.
//! When streaming is asked for, the sink still receives a
//! bracketing `queued` + `gen_start` + `gen_done` progress trio
//! plus a single chunk with the full text so wire consumers that
//! expect any events don't stall.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chitin_llama::{LoadConfig, Session};
use gemma_utils::{ChatFormat, Gemma4Format, Gemma4ToolFormat, ToolFormat};
use model_api_proto::{
    InferenceInput, InferenceResponse, ToolMode,
};

use crate::slot::{SlotHandle, SlotRequest, SlotResponse, StreamSink};
use crate::{tool_text, turn_render};

/// SlotHandle backed by a `chitin_llama::Session`.
///
/// The session isn't `Send + Sync` on its own (holds a raw
/// llama.cpp context pointer), so we own it exclusively behind an
/// `Arc<Mutex<>>` and drive all inference under `smol::unblock` so
/// the FFI stays off the async executor's carrier threads.
pub struct LlamaSlot {
    session: Arc<Mutex<Session>>,
    model_name: String,
    /// Active tool-format for marker extraction (`Client` mode).
    tool_format: Arc<dyn ToolFormat>,
    /// Active chat template for `InferenceInput::Turns` rendering.
    chat_format: Arc<dyn ChatFormat>,
}

impl LlamaSlot {
    /// Load a GGUF and hand back a ready-to-serve slot. `model_dir`
    /// may be either a `.gguf` file (used as-is) or a directory
    /// containing one (first sorted `.gguf` wins). `gguf_path`
    /// explicitly overrides both.
    ///
    /// Gemma 4 is the current production target; `tool_format` +
    /// `chat_format` are hard-coded to Gemma-4 for now. A future
    /// CLI flag can pick the pair for other model families.
    pub fn spawn(
        model_dir: String,
        gguf_path: Option<String>,
        model_name: String,
        _max_tokens: usize,
        max_seq_len: u32,
    ) -> Result<Self, String> {
        let gguf_path = gguf_path.unwrap_or_else(|| resolve_gguf_path(&model_dir));
        let load_cfg = LoadConfig {
            gguf_path,
            max_seq_len,
            n_gpu_layers: -1,
            // See parent workspace's llama_slot.rs comment for the
            // 128 rationale: at deep n_ctx=32K, the per-layer
            // attention-scores intermediate scales with
            // `n_batch × n_ctx × n_heads × 2B` — 512 pushed the M4's
            // Metal budget into IOGPUCommandBufferCallbackError.
            n_batch: 128,
            draft_gguf_path: None,
            mtp_draft_n: 0,
        };
        let session = Session::load(load_cfg)?;
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            model_name,
            tool_format: Arc::new(Gemma4ToolFormat),
            chat_format: Arc::new(Gemma4Format),
        })
    }
}

#[async_trait]
impl SlotHandle for LlamaSlot {
    fn model_name(&self) -> &str { &self.model_name }
    fn gpu_memory_mb(&self) -> Option<u32> { None }

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
    async fn run_inner(
        &self,
        req: SlotRequest,
        sink: Option<&dyn StreamSink>,
    ) -> Result<SlotResponse, String> {
        let SlotRequest { req } = req;

        let want_client_dispatch = match req.inference_config.tool_mode {
            ToolMode::Client => true,
            ToolMode::Server => false,
            ToolMode::Auto => !req.tools.is_empty(),
        };

        // Prompt assembly. Two paths matching the wire input variant:
        //  - Turns: render_turns applies the chat template to a full
        //    turn list; tool_results_to_turns splices any incoming
        //    ToolResults in as Role::Tool turns first.
        //  - Text: raw string. Tool results (if any) get prepended
        //    to the text as pre-templated blocks.
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

        // Basic progress trio so streaming consumers see start/end
        // events even though we don't yet stream individual chunks.
        if let Some(s) = sink {
            s.on_progress(model_api_proto::ProgressEvent {
                phase: "queued".into(),
                tool: None,
                detail: None,
            });
            s.on_progress(model_api_proto::ProgressEvent {
                phase: "gen_start".into(),
                tool: None,
                detail: None,
            });
        }

        // Blocking inference on the smol blocking pool so the executor
        // carrier threads stay free. Session's mutex is held only
        // inside the closure — never crossed .await.
        let session = self.session.clone();
        let max_tokens = req.max_tokens as usize;
        let text = smol::unblock(move || -> Result<String, String> {
            let mut sess = session.lock().map_err(|e| format!("session mutex poisoned: {e}"))?;
            let prompt_tokens = sess.tokenize(&prompt, true)?;
            let out_tokens = sess.generate_raw(&prompt_tokens, max_tokens)?;
            Ok(sess.detokenize(&out_tokens))
        }).await?;

        if let Some(s) = sink {
            // Single "everything" chunk. When we grow real streaming
            // this splits into per-token chunks and the phase order
            // becomes queued → gen_start → chunk×N → gen_done.
            s.on_chunk(model_api_proto::StreamChunk {
                delta_text: text.clone(),
                finish_reason: None,
                phase: None,
            });
            s.on_progress(model_api_proto::ProgressEvent {
                phase: "gen_done".into(),
                tool: None,
                detail: None,
            });
        }

        let tool_calls = if want_client_dispatch {
            tool_text::extract_tool_calls(&text, &*self.tool_format)
        } else {
            Vec::new()
        };

        // session_id echoing: Persistent sessions get their id
        // reflected back. Non-persistent (Stateless) returns None so
        // clients can distinguish. Prefix-cache continuity for
        // Persistent sessions is a follow-up; today every request
        // re-prefills the full prompt.
        let session_id = match req.session {
            model_api_proto::SessionMode::Persistent { session_id } => Some(session_id),
            model_api_proto::SessionMode::Stateless => None,
        };

        Ok(SlotResponse(InferenceResponse {
            text,
            session_id,
            raw_text: None,
            injections: Vec::new(),
            tool_calls,
            replacement: None,
        }))
    }
}

/// Resolve a caller's `--model` argument to a concrete `.gguf`
/// path. Matches the parent workspace's `resolve_gguf_path` in
/// `thinker_impl::llama_slot`: extension takes precedence over
/// filesystem existence so `/path/to/x.gguf` is honoured before
/// the model is fetched; a directory input picks its first
/// sorted `.gguf`; anything else falls back to the historical
/// `<dir>/model.gguf` symlink.
fn resolve_gguf_path(model_dir: &str) -> String {
    let p = std::path::Path::new(model_dir);
    let is_gguf_name = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false);
    if is_gguf_name {
        return model_dir.to_string();
    }
    if p.is_dir() {
        if let Ok(rd) = std::fs::read_dir(p) {
            let mut ggufs: Vec<std::path::PathBuf> = rd
                .flatten()
                .map(|e| e.path())
                .filter(|q| {
                    q.extension()
                        .and_then(|e| e.to_str())
                        .map(|s| s.eq_ignore_ascii_case("gguf"))
                        .unwrap_or(false)
                })
                .collect();
            ggufs.sort();
            if let Some(first) = ggufs.first() {
                log::info!(
                    "llama_slot: picked {} from {} (set gguf_path to override)",
                    first.display(),
                    p.display(),
                );
                return first.to_string_lossy().into_owned();
            }
        }
    }
    format!("{model_dir}/model.gguf")
}
