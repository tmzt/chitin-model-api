//! `SlotHandle` backed by the `litertlm` crate (Google LiteRT-LM
//! Rust bindings to the C engine API). Used on the Pixel demo to
//! run Gemma-4 (E2B / E4B) on the PowerVR GPU via OpenCL — the
//! path that llama.cpp's Vulkan + OpenCL backends can't handle.
//!
//! Two-layer state model:
//! - **Engine** lives for the server's lifetime; one model load.
//! - **Conversation** (a `litert_lm_conversation_*` C handle) is
//!   the unit of KV-cache state. Same C ptr reused across
//!   `send_message_stream` calls → history accumulates
//!   (deps/litert-rs/litert-lm/src/conversation.rs:101-206).
//!   This slot pools Conversations by `SessionMode::Persistent`
//!   `session_id` so multi-turn voice conversations don't re-prefill
//!   the whole transcript every turn.
//!
//! Pool sizing: `--litertlm-session-pool-size N` (LRU-ish, count-
//! based) + `--litertlm-session-ttl-secs S` (treat as cold-miss if
//! last_used is older).
//!
//! Templating: the C `Conversation` applies the model's
//! `chat_template.jinja` from the .litertlm file internally. The
//! slot does NOT pre-render anything — `InferenceInput::Turns` are
//! fed one `send_message_stream` per turn; `InferenceInput::Text`
//! is treated as a single user-role turn.
//!
//! Cold-miss replay caveat: when a long transcript arrives without
//! a matching pool entry (server restart between turns), we replay
//! user turns by calling `send_message_stream` for each one and
//! discarding all but the final response. The Conversation's
//! internal KV cache then holds the user turns interleaved with
//! whatever assistant text the model regenerated — not the
//! original assistant turns. For short demo conversations this is
//! tolerable; for long sessions across restarts we'd want
//! `litert_lm_conversation_config_set_messages_json` (not yet
//! exposed in the safe wrapper).
//!
//! Tool calls: not wired. LiteRT-LM has native tool support via
//! `set_tools` on the conversation config; that lands in a
//! follow-up. The slot returns `tool_calls: Vec::new()` today.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use litertlm::{Backend, Conversation, Engine, EngineSettings, SamplerParams};
use model_api_proto::{InferenceInput, InferenceResponse, Role, SessionMode, StreamChunk, Turn};

use crate::slot::{SlotHandle, SlotRequest, SlotResponse, StreamSink};

/// One pooled session. The `Conversation` is sync (its
/// `send_message_stream` takes `&mut self`), so we wrap it in a
/// `Mutex` so two requests for the same `session_id` serialise.
struct Pooled {
    conv: Conversation,
    /// Number of turns from the wire's `Turns(...)` payload that
    /// have already been sent into this Conversation. The next
    /// request only needs to send `turns[sent_turns..]`. Reset to
    /// `turns.len()` after each successful send.
    sent_turns: usize,
    last_used: Instant,
}

pub struct LiteRtLmSlot {
    engine: Arc<Engine>,
    model_name: String,
    /// CLI-supplied default sampler params. Per-request inference
    /// config overrides these in `sampler_for_request`. NOTE: the
    /// sampler is baked into the Conversation at creation time;
    /// changes after first request on a session_id are ignored.
    default_sampler: SamplerParams,
    /// Per-Conversation visual token budget passed via
    /// `LiteRtLmConversationOptionalArgs`. Multimodal Gemma-4
    /// models reject text-only sends without it. `None` -> rely on
    /// model defaults (works for pure-text models).
    visual_token_budget: Option<i32>,
    /// Conversation pool. Outer `Mutex` for insert/evict; inner
    /// `Mutex<Pooled>` so concurrent requests on different
    /// session_ids don't block each other.
    sessions: Mutex<HashMap<String, Arc<Mutex<Pooled>>>>,
    pool_size: usize,
    ttl: Duration,
}

impl LiteRtLmSlot {
    pub fn new(
        model_path: PathBuf,
        model_name: String,
        backend: Backend,
        max_num_tokens: i32,
        visual_token_budget: Option<i32>,
        pool_size: usize,
        ttl: Duration,
    ) -> Result<Self, String> {
        let settings = EngineSettings::new(model_path)
            .backend(backend)
            .max_num_tokens(max_num_tokens);
        let engine = Engine::new(settings)
            .map_err(|e| format!("litertlm Engine::new: {e}"))?;
        // v0.13's sampler_factory only implements TYPE_UNSPECIFIED
        // (no-op) and TOP_P; TopK and Greedy return UnimplementedError.
        // SamplerParams::default() picks TopK, which would blow up,
        // so override to TopP up front.
        let default_sampler = SamplerParams::default().top_p(0.95);
        Ok(Self {
            engine: Arc::new(engine),
            model_name,
            default_sampler,
            visual_token_budget,
            sessions: Mutex::new(HashMap::new()),
            pool_size: pool_size.max(1),
            ttl,
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

    /// Normalise the wire input into a list of (turns_to_send,
    /// sent_turns_after) where turns_to_send is the slice the
    /// caller should feed to the Conversation. For `Text(s)` this
    /// is always a single User turn. For `Turns(ts)` with a
    /// matching pool entry, only the suffix past `sent_turns`.
    fn slice_to_send<'a>(turns: &'a [Turn], already_sent: usize) -> &'a [Turn] {
        if already_sent >= turns.len() {
            &[]
        } else {
            &turns[already_sent..]
        }
    }

    /// Build a fresh Conversation for either a Stateless request or
    /// a cold-miss Persistent one.
    fn make_conv(&self, sampler: SamplerParams) -> Result<Conversation, String> {
        let mut conv = self
            .engine
            .create_conversation(sampler)
            .map_err(|e| format!("litertlm create_conversation: {e}"))?;
        if let Some(b) = self.visual_token_budget {
            conv = conv.with_visual_token_budget(b);
        }
        Ok(conv)
    }

    /// Get-or-create the pool entry for `id`. On cold miss, evicts
    /// the oldest entry if the pool is at capacity. TTL-expired
    /// entries are treated as cold-miss (the stale Conversation is
    /// dropped and a fresh one is created).
    fn get_or_create_session(
        &self,
        id: &str,
        sampler: SamplerParams,
    ) -> Result<(Arc<Mutex<Pooled>>, bool), String> {
        let mut pool = self.sessions.lock().unwrap();

        // Cache hit (and not TTL-expired)?
        if let Some(p) = pool.get(id) {
            let mut g = p.lock().unwrap();
            if g.last_used.elapsed() < self.ttl {
                g.last_used = Instant::now();
                let ptr = Arc::clone(p);
                drop(g);
                return Ok((ptr, true));
            }
            // TTL-expired — drop the stale entry and fall through to
            // the fresh-create path. dropping inner Mutex guard
            // first, then the outer remove below.
            drop(g);
            pool.remove(id);
        }

        // Cold miss. Evict the oldest if we're at capacity.
        if pool.len() >= self.pool_size {
            let oldest = pool
                .iter()
                .min_by_key(|(_, p)| p.lock().unwrap().last_used)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest {
                log::info!("[litertlm] evicting session '{k}' (pool full, size={})", self.pool_size);
                pool.remove(&k);
            }
        }

        log::info!("[litertlm] cold-miss session '{id}' — creating Conversation");
        let conv = self.make_conv(sampler)?;
        let pooled = Arc::new(Mutex::new(Pooled {
            conv,
            sent_turns: 0,
            last_used: Instant::now(),
        }));
        pool.insert(id.to_string(), Arc::clone(&pooled));
        Ok((pooled, false))
    }
}

/// Turn list to feed into a Conversation: extracts the User /
/// Tool turns from a Turn slice (Assistant turns are skipped
/// because the Conversation regenerates them; System turns are
/// folded into the first user message by the chat template).
/// Returns the content strings in order.
fn user_facing_contents(turns: &[Turn]) -> Vec<String> {
    turns
        .iter()
        .filter_map(|t| match t.role {
            Role::User | Role::Tool => Some(t.content.clone()),
            // System belongs in the conversation's system_message
            // config slot (not yet wired); for now we just drop it.
            // Assistant turns are skipped — see cold-miss caveat in
            // the module docs.
            Role::System | Role::Assistant => None,
        })
        .collect()
}

/// Reduce a wire request to (turns_to_send_contents, full_turn_count).
/// `full_turn_count` is what we'll set `sent_turns` to after a
/// successful dispatch — counts the original wire-supplied turn
/// list length so the next request can compute its delta.
fn plan_dispatch(req: &SlotRequest, already_sent: usize) -> Result<(Vec<String>, usize), String> {
    match &req.req.input {
        InferenceInput::Text(s) => {
            // Text is a single untemplated user message. We can't
            // efficiently compute "delta" for Text callers, so we
            // always send exactly this one message and bump
            // sent_turns by 1.
            Ok((vec![s.clone()], already_sent + 1))
        }
        InferenceInput::Turns(turns) => {
            let suffix = LiteRtLmSlot::slice_to_send(turns, already_sent);
            let contents = user_facing_contents(suffix);
            if contents.is_empty() {
                return Err(
                    "Turns: no user-facing turns to send (Assistant-only suffix?)".into(),
                );
            }
            Ok((contents, turns.len()))
        }
        InferenceInput::Pcm { .. } | InferenceInput::Mel { .. } => {
            Err("audio inputs not supported on litert-lm backend".into())
        }
    }
}

/// Build the SlotResponse from a final text. The litert-lm slot
/// returns `tool_calls` empty until we wire `set_tools` to the
/// Conversation; raw_text / injections / replacement aren't
/// produced on this backend.
fn build_response(text: String, session_id: Option<String>) -> SlotResponse {
    SlotResponse(InferenceResponse {
        text,
        session_id,
        raw_text: None,
        injections: Vec::new(),
        tool_calls: Vec::new(),
        replacement: None,
    })
}

#[async_trait]
impl SlotHandle for LiteRtLmSlot {
    fn model_name(&self) -> &str {
        &self.model_name
    }
    fn gpu_memory_mb(&self) -> Option<u32> {
        None
    }

    async fn run(&self, req: SlotRequest) -> Result<SlotResponse, String> {
        // Non-streaming = streaming with a discarding sink. Both
        // paths funnel through send_message_stream so the
        // conversation state (KV cache) advances identically.
        self.run_stream(req, &crate::slot::DiscardSink).await
    }

    async fn run_stream(
        &self,
        req: SlotRequest,
        sink: &dyn StreamSink,
    ) -> Result<SlotResponse, String> {
        let sampler = self.sampler_for_request(&req);
        let session_id = match &req.req.session {
            SessionMode::Persistent { session_id } => Some(session_id.clone()),
            SessionMode::Stateless => None,
        };

        // Resolve pool entry: stateless → fresh Conversation (not
        // pooled, dropped after the request); persistent → pool
        // get-or-create.
        let pooled = match &session_id {
            Some(id) => {
                let (p, hit) = self.get_or_create_session(id, sampler.clone())?;
                log::debug!("[litertlm] session '{id}' {}", if hit { "HIT" } else { "MISS" });
                Some(p)
            }
            None => None,
        };

        // Compute what to actually send.
        let (contents, new_sent_turns) = {
            let already_sent = pooled
                .as_ref()
                .map(|p| p.lock().unwrap().sent_turns)
                .unwrap_or(0);
            plan_dispatch(&req, already_sent)?
        };

        // bounded(64) backpressures the C callback when the wire
        // can't keep up with token emission.
        let (tx, rx) = async_channel::bounded::<String>(64);

        // The generator task either drives the pooled Conversation
        // or a stateless one; in both cases it sends each turn's
        // chunks into `tx` and returns the FINAL turn's full text.
        let engine = self.engine.clone();
        let vtb = self.visual_token_budget;
        let pooled_for_task = pooled.clone();
        let gen_task = smol::unblock(move || {
            // Local helper that runs a list of user messages
            // through whatever Conversation is on hand, sending
            // each chunk into tx and returning the LAST message's
            // accumulated response.
            let send_all = |conv: &mut Conversation, msgs: &[String]| -> Result<String, String> {
                let mut last_response = String::new();
                for (i, msg) in msgs.iter().enumerate() {
                    let is_last = i + 1 == msgs.len();
                    let mut acc = String::new();
                    conv.send_message_stream(msg, |chunk: &str| {
                        if is_last {
                            // Only stream the final turn's chunks
                            // to the wire — intermediate replay
                            // chunks are debug noise.
                            let _ = smol::block_on(tx.send(chunk.to_string()));
                        }
                        acc.push_str(chunk);
                    })
                    .map_err(|e| format!("litertlm send_message_stream: {e}"))?;
                    if is_last {
                        last_response = acc;
                    }
                }
                Ok(last_response)
            };

            let text = if let Some(p) = pooled_for_task {
                let mut g = p.lock().unwrap();
                let r = send_all(&mut g.conv, &contents)?;
                g.sent_turns = new_sent_turns;
                g.last_used = Instant::now();
                r
            } else {
                // Stateless: ephemeral Conversation, dropped after.
                let mut conv = {
                    let mut c = engine
                        .create_conversation(sampler)
                        .map_err(|e| format!("litertlm create_conversation: {e}"))?;
                    if let Some(b) = vtb {
                        c = c.with_visual_token_budget(b);
                    }
                    c
                };
                send_all(&mut conv, &contents)?
            };
            drop(tx);
            Ok::<String, String>(text)
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

        // Final stop marker so consumers can stop draining without
        // waiting for InferenceComplete on the wire.
        sink.on_chunk(StreamChunk {
            delta_text: String::new(),
            finish_reason: Some("stop".into()),
            phase: Some("text".into()),
        });

        Ok(build_response(text, session_id))
    }
}
