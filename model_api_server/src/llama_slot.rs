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
use common::tool_format::{Gemma4ToolFormat, ToolFormat};
use common::types::{ExternalToolDef, ExternalToolParam};
use model_api_proto::{
    GpuRole, InferenceInput, InferenceRequest, InferenceResponse, JsonMode, SessionMode,
    SessionReplacement, ToolCall, ToolDef, ToolMode, ToolResult,
};
use thinker_impl::ThinkerConfig;

use crate::slot::{SlotHandle, SlotRequest, SlotResponse};

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
    /// [`extract_tool_calls`] to parse markers out of `raw_text`
    /// when the wire request asked for client-side tool dispatch.
    tool_format: Arc<dyn ToolFormat>,
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
        // ToolFormat is the abstraction point for other models. To
        // support a different model, add a CLI flag that picks the
        // right ToolFormat impl here (and adjust the channels
        // driver's marker detection — same Arc<dyn ToolFormat>
        // value).
        let tool_format: Arc<dyn ToolFormat> = Arc::new(Gemma4ToolFormat);

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
        Ok(Self { resource, tool_format })
    }
}

#[async_trait]
impl SlotHandle for LlamaSlot {
    fn model_name(&self) -> &str { self.resource.model_name() }
    fn gpu_memory_mb(&self) -> Option<u32> { self.resource.gpu_memory_mb() }

    async fn run(&self, req: SlotRequest) -> Result<SlotResponse, String> {
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

        // Fold any prior turn's tool results into the user input.
        // Each ToolResult is formatted via the active model's
        // ToolFormat (Gemma 4: `<tool_response>...</tool_response>`)
        // and prepended to the new user message so the model sees
        // its prior `<tool_call>` echoed back with the result.
        let prompt = build_prompt_with_tool_results(
            &req.input, &req.tool_results, &*self.tool_format,
        )?;

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

        let thinker_req = ThinkerRequest {
            role: to_common_role(req.role),
            input: CommonInput::Text(prompt),
            max_tokens: req.max_tokens as usize,
            response_tx: resp_tx,
            session: to_common_session(req.session),
            stream_tx: None,
            progress_tx: None,
            prefix_cache: None,
            disable_think_prefix: req.inference_config.disable_think_prefix,
            inference_config: common_cfg,
            cache_hash: req.cache_hash,
        };

        let tx = self.resource.request_sender();
        if let Err(e) = tx.send(thinker_req).await {
            return Err(format!("slot channel closed: {e}"));
        }

        let common_resp = resp_rx.recv().await
            .map_err(|e| format!("slot response channel closed: {e}"))?;

        // Extract tool calls from the raw model output when we're in
        // client-dispatch mode. The channels driver leaves markers
        // verbatim in `text` (raw_tool_call short-circuits the
        // marker-stripping state machine), so we scan with the
        // active model's ToolFormat.
        let tool_calls = if want_client_dispatch {
            extract_tool_calls(&common_resp.text, &*self.tool_format)
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

// ── Tool result folding + tool call extraction ──────────────────────

/// Build the user input for the next turn. When prior tool results
/// are present, each is wrapped via the active model's
/// `ToolFormat::format_response` and prepended in order; the
/// original `input` text becomes the trailing user message.
///
/// `tool_results` is empty on the very first turn of a tool-using
/// session, in which case this is a no-op pass-through. `call_id`
/// is informational only today (Gemma 4's format ignores it);
/// retained on the wire so future formats that need correlation
/// can use it.
fn build_prompt_with_tool_results(
    input: &InferenceInput,
    tool_results: &[ToolResult],
    tool_format: &dyn ToolFormat,
) -> Result<String, String> {
    let user_text = match input {
        InferenceInput::Text(t) => t.clone(),
        InferenceInput::Pcm { .. } | InferenceInput::Mel { .. } => {
            return Err("audio inputs not supported on the llama backend".into());
        }
    };
    if tool_results.is_empty() {
        return Ok(user_text);
    }
    let mut out = String::new();
    for r in tool_results {
        out.push_str(&tool_format.format_response(&r.output));
        out.push('\n');
    }
    out.push_str(&user_text);
    Ok(out)
}

/// Scan `text` for tool-call marker pairs and convert each body
/// into a wire [`ToolCall`]. Uses `tool_format.open_marker()` /
/// `close_marker()` to locate pairs and `parse_body` to extract
/// `(name, args)`. Tool ids are positional (`tc-0`, `tc-1`, …) —
/// the underlying ToolFormat doesn't surface ids; clients should
/// echo whatever id they received in the matching `ToolResult`.
pub(crate) fn extract_tool_calls(text: &str, tool_format: &dyn ToolFormat) -> Vec<ToolCall> {
    let open = tool_format.open_marker();
    let close = tool_format.close_marker();
    if open.is_empty() || close.is_empty() {
        return Vec::new();
    }
    let mut calls = Vec::new();
    let mut cursor = 0usize;
    while let Some(start) = text[cursor..].find(open) {
        let body_start = cursor + start + open.len();
        let Some(end) = text[body_start..].find(close) else { break };
        let body = &text[body_start..body_start + end];
        cursor = body_start + end + close.len();
        if let Some(parsed) = tool_format.parse_body(body) {
            let args_json = serde_json::to_string(&parsed.args)
                .unwrap_or_else(|_| "{}".to_string());
            calls.push(ToolCall {
                id: format!("tc-{}", calls.len()),
                name: parsed.name,
                arguments_json: args_json,
            });
        }
    }
    calls
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

// ── Unit tests for the parser ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use common::tool_format::Gemma4ToolFormat;

    #[test]
    fn extract_zero_calls_from_plain_text() {
        let f = Gemma4ToolFormat;
        let calls = extract_tool_calls("the answer is 4.", &f);
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn extract_one_call_gemma4_shape() {
        let f = Gemma4ToolFormat;
        // Gemma 4 emits `<|tool_call>call:NAME{ARGS}<tool_call|>`.
        let text = "reasoning… <|tool_call>call:calculator{\"expr\":\"2+2\"}<tool_call|> done";
        let calls = extract_tool_calls(text, &f);
        assert_eq!(calls.len(), 1, "expected 1 call, got {calls:?}");
        assert_eq!(calls[0].name, "calculator");
        assert_eq!(calls[0].id, "tc-0");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments_json).unwrap();
        assert_eq!(args.get("expr").and_then(|v| v.as_str()), Some("2+2"));
    }

    #[test]
    fn extract_multiple_calls_keep_order_and_unique_ids() {
        let f = Gemma4ToolFormat;
        let text = "\
            <|tool_call>call:a{\"k\":1}<tool_call|>\n\
            interlude\n\
            <|tool_call>call:b{\"k\":2}<tool_call|>";
        let calls = extract_tool_calls(text, &f);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].id, "tc-0");
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].id, "tc-1");
    }
}
