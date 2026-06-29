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
//!
//! ## Prefix cache (replaces the per-session pool)
//!
//! Conversations are pooled by a **content hash of the conversation
//! prefix** rather than by wire `session_id`. The hash inputs are
//! the canonicalized (system_prompt + sampler config + turns[..N])
//! bytes; the lookup is "longest prefix that's in the cache."
//!
//! When the hash hits, we use `Conversation::clone()`
//! (KV-state deep copy, `litert_lm_conversation_clone` under the
//! hood) to derive a working copy off the cached entry. That copy
//! is advanced with the suffix turns and re-cached under the new
//! full-conversation hash on completion.
//!
//! Two maps:
//! - `prefix_cache: HashMap<u64, CachedConv>` — keyed by the upper
//!   64 bits of `xxh3_128(canonical_bytes)`. Mirrors
//!   `common::util::xxhash3_hex16` so callers on either side of a
//!   process boundary can share fingerprints.
//! - `sessions: HashMap<String, u64>` — `wire session_id → last
//!   known full-conversation hash`. Lets consecutive turns on the
//!   same session_id short-circuit the longest-prefix scan with a
//!   single direct lookup. Pure perf — correctness is preserved
//!   even if `sessions` is empty.
//!
//! ## Cold-start replay drift
//!
//! On a partial prefix hit we replay `turns[N..]`'s User/Tool
//! messages via `send_message_stream` one at a time. Assistant
//! turns embedded in the suffix are SKIPPED — the model regenerates
//! them in place, so the resulting KV cache holds the model's fresh
//! assistant text rather than the original. Only the FINAL replay
//! turn's chunks are forwarded to the wire sink; intermediate
//! regenerated responses are accumulated locally and dropped to
//! keep wire output clean. For short transcripts this drift is
//! invisible; for long sessions where the suffix contains older
//! assistant turns we'd want
//! `litert_lm_conversation_config_set_messages_json` (not yet
//! exposed by the safe wrapper).
//!
//! Cache sizing: `--litertlm-prefix-cache-size N` (LRU by
//! `last_used: Instant`, count-based) +
//! `--litertlm-prefix-cache-ttl-secs S` (TTL-expired entries are
//! treated as cold-miss). Old `--litertlm-session-pool-size` /
//! `--litertlm-session-ttl-secs` flags still accepted as deprecated
//! aliases so chitin-stack.sh keeps working through the rename.
//!
//! Templating: the C `Conversation` applies the model's
//! `chat_template.jinja` from the .litertlm file internally. The
//! slot does NOT pre-render anything — `InferenceInput::Turns` are
//! fed one `send_message_stream` per turn; `InferenceInput::Text`
//! is treated as a single user-role turn.
//!
//! Tool calls: not wired. LiteRT-LM has native tool support via
//! `set_tools` on the conversation config; that lands in a
//! follow-up. The slot returns `tool_calls: Vec::new()` today.
//! Tool-role *result* turns are hashed as their own role (3) so
//! they stay distinguishable from plain User turns in the cache
//! key; the actual send still treats them as user-formatted
//! messages (the tool_format wraps their content upstream).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use litertlm::{Backend, Conversation, Engine, EngineSettings, SamplerParams};
use model_api_proto::{InferenceInput, InferenceResponse, Role, SessionMode, StreamChunk, Turn};
use xxhash_rust::xxh3::xxh3_128;

use crate::slot::{SlotHandle, SlotRequest, SlotResponse, StreamSink};

/// Default top-p when the request doesn't set one. Must match the
/// value baked into `default_sampler` below so hash and actual
/// sampler stay aligned.
const DEFAULT_TOP_P: f32 = 0.95;
/// Default temperature when the request doesn't set one. Matches
/// `SamplerParams::default()`'s baked-in 0.8.
const DEFAULT_TEMPERATURE: f32 = 0.8;

/// One pooled prefix. The `Conversation` is the cache's "donor" —
/// we never call `send_message_stream` on it directly; we
/// [`Conversation::clone`] it to make a working copy and advance
/// the clone. That keeps the cached KV state pristine and
/// shareable across concurrent requests with the same prefix.
struct CachedConv {
    conv: Conversation,
    /// Number of turns this entry's hash represents. Used only for
    /// log lines and the session-pin sanity check (we compare
    /// `c.turn_count` against `turns.len()` before trusting it as a
    /// prefix anchor).
    turn_count: usize,
    last_used: Instant,
}

pub struct LiteRtLmSlot {
    engine: Arc<Engine>,
    model_name: String,
    /// CLI-supplied default sampler params. Per-request inference
    /// config overrides these in `sampler_for_request`. The sampler
    /// is baked into the Conversation at creation time and is part
    /// of the prefix-cache key — a sampler change yields a fresh
    /// cache entry rather than silently reusing a stale one.
    default_sampler: SamplerParams,
    /// Per-Conversation visual token budget passed via
    /// `LiteRtLmConversationOptionalArgs`. Multimodal Gemma-4
    /// models reject text-only sends without it. `None` -> rely on
    /// model defaults (works for pure-text models).
    visual_token_budget: Option<i32>,
    /// Content-hash-keyed prefix cache. `Arc` so the dispatch task
    /// (running on `smol::unblock`'s blocking pool) can share
    /// ownership with the async slot handle.
    prefix_cache: Arc<Mutex<HashMap<u64, CachedConv>>>,
    /// `wire session_id → last-known full-conversation hash`. Bumps
    /// the cache lookup from O(turns.len()) hashes to O(1) in the
    /// common case where successive requests reuse a session_id.
    sessions: Arc<Mutex<HashMap<String, u64>>>,
    cache_size: usize,
    ttl: Duration,
}

impl LiteRtLmSlot {
    pub fn new(
        model_path: PathBuf,
        model_name: String,
        backend: Backend,
        max_num_tokens: i32,
        visual_token_budget: Option<i32>,
        cache_size: usize,
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
        // so override to TopP up front. Must match DEFAULT_TOP_P /
        // DEFAULT_TEMPERATURE so hash inputs line up with what the
        // C side actually samples with.
        let default_sampler = SamplerParams::default().top_p(DEFAULT_TOP_P);
        Ok(Self {
            engine: Arc::new(engine),
            model_name,
            default_sampler,
            visual_token_budget,
            prefix_cache: Arc::new(Mutex::new(HashMap::new())),
            sessions: Arc::new(Mutex::new(HashMap::new())),
            cache_size: cache_size.max(1),
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

    /// The (top_p, temperature) tuple that will actually be baked
    /// into the Conversation, mirroring `sampler_for_request`'s
    /// override chain. We fingerprint this — not the
    /// `SamplerParams` struct directly — because the struct's
    /// fields are `pub(crate)` in the litertlm crate.
    fn effective_sampler_for_hash(req: &SlotRequest) -> (f32, f32) {
        let top_p = req
            .req
            .inference_config
            .top_p
            .unwrap_or(DEFAULT_TOP_P);
        let temp = req
            .req
            .inference_config
            .temperature
            .unwrap_or(DEFAULT_TEMPERATURE);
        (top_p, temp)
    }
}

/// Canonical byte serialization for prefix-cache hashing.
///
/// Layout (cross-checked with the litert-rs fork agent; both sides
/// MUST produce identical bytes for the same logical prefix):
/// ```text
///   "sys\0"            (4 bytes literal)
///   <system_prompt bytes>     (empty if None — no length prefix)
///   "\0sampler\0"      (9 bytes literal)
///   <top_p as f32 LE>  (4 bytes)
///   <temperature as f32 LE> (4 bytes)
///   "\0turns\0"        (7 bytes literal)
///   for each turn:
///     <role_byte>          (1 byte: 0=System, 1=User, 2=Assistant, 3=Tool)
///     <content len as u32 LE>   (4 bytes)
///     <content bytes>           (content.len() bytes)
///     if turn.tool_call_id is Some:
///       <tool_call_id len as u32 LE>   (4 bytes)
///       <tool_call_id bytes>           (id.len() bytes)
///     (omitted entirely when None — role discriminator + Turn
///     constructor convention keeps this unambiguous: only Tool
///     turns carry a tool_call_id in our schema.)
/// ```
fn canonical_prefix_bytes(
    system_prompt: Option<&str>,
    sampler_top_p: f32,
    sampler_temperature: f32,
    turns: &[Turn],
) -> Vec<u8> {
    // Conservative capacity hint: literal delimiters + sampler bytes +
    // per-turn overhead (role + length prefixes) + content payloads.
    let payload: usize = turns
        .iter()
        .map(|t| {
            1 + 4
                + t.content.len()
                + t.tool_call_id.as_ref().map_or(0, |x| 4 + x.len())
        })
        .sum();
    let mut buf = Vec::with_capacity(64 + system_prompt.map_or(0, |s| s.len()) + payload);

    buf.extend_from_slice(b"sys\0");
    if let Some(s) = system_prompt {
        buf.extend_from_slice(s.as_bytes());
    }
    buf.extend_from_slice(b"\0sampler\0");
    buf.extend_from_slice(&sampler_top_p.to_le_bytes());
    buf.extend_from_slice(&sampler_temperature.to_le_bytes());
    buf.extend_from_slice(b"\0turns\0");
    for t in turns {
        let role_byte: u8 = match t.role {
            Role::System => 0,
            Role::User => 1,
            Role::Assistant => 2,
            Role::Tool => 3,
        };
        buf.push(role_byte);
        let content_bytes = t.content.as_bytes();
        let clen = u32::try_from(content_bytes.len()).unwrap_or(u32::MAX);
        buf.extend_from_slice(&clen.to_le_bytes());
        buf.extend_from_slice(content_bytes);
        if let Some(tcid) = &t.tool_call_id {
            let tcid_bytes = tcid.as_bytes();
            let tlen = u32::try_from(tcid_bytes.len()).unwrap_or(u32::MAX);
            buf.extend_from_slice(&tlen.to_le_bytes());
            buf.extend_from_slice(tcid_bytes);
        }
    }
    buf
}

/// Compute the prefix-cache key for `turns[..]`. Takes the upper 64
/// bits of `xxh3_128`, mirroring `common::util::xxhash3_hex16`.
fn hash_prefix(
    system_prompt: Option<&str>,
    sampler_top_p: f32,
    sampler_temperature: f32,
    turns: &[Turn],
) -> u64 {
    let bytes = canonical_prefix_bytes(system_prompt, sampler_top_p, sampler_temperature, turns);
    let h = xxh3_128(&bytes);
    (h >> 64) as u64
}

/// Extract the canonical turn list from the wire input. `Text(s)`
/// is treated as a single User turn; `Pcm`/`Mel` aren't supported
/// on this backend.
fn input_to_turns(input: &InferenceInput) -> Result<Vec<Turn>, String> {
    match input {
        InferenceInput::Text(s) => Ok(vec![Turn::user(s.clone())]),
        InferenceInput::Turns(t) => Ok(t.clone()),
        InferenceInput::Pcm { .. } | InferenceInput::Mel { .. } => {
            Err("audio inputs not supported on litert-lm backend".into())
        }
    }
}

/// Filter a turn slice down to the content strings that should be
/// sent to the Conversation via `send_message_stream`. Assistant
/// turns are skipped (the Conversation regenerates them); System
/// turns are dropped today (no `set_system_message` wiring yet on
/// this path); Tool result turns become user-formatted messages —
/// their content is already wrapped by `common::tool_format` before
/// it reaches us, so we just forward as-is.
fn sendable_contents(turns: &[Turn]) -> Vec<String> {
    turns
        .iter()
        .filter_map(|t| match t.role {
            Role::User | Role::Tool => Some(t.content.clone()),
            Role::System | Role::Assistant => None,
        })
        .collect()
}

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
        let (top_p_h, temp_h) = Self::effective_sampler_for_hash(&req);

        let session_id = match &req.req.session {
            SessionMode::Persistent { session_id } => Some(session_id.clone()),
            SessionMode::Stateless => None,
        };

        let system_prompt = req.req.inference_config.system_prompt.clone();
        let turns: Vec<Turn> = input_to_turns(&req.req.input)?;

        if turns.is_empty() {
            return Err("litertlm: empty input (no turns to dispatch)".into());
        }

        // Resolve the session pin up front (outside the blocking
        // closure) so we can pass a plain Option<u64> through and
        // avoid nested-lock pitfalls inside the cache critical
        // section.
        let session_pin = session_id.as_deref().and_then(|id| {
            self.sessions.lock().unwrap().get(id).copied()
        });

        // Bounded channel for streaming chunks from the C callback
        // to the wire sink. 64 keeps the wire side from buffering
        // huge bursts when the consumer stalls.
        let (tx, rx) = async_channel::bounded::<String>(64);

        let prefix_cache = self.prefix_cache.clone();
        let sessions = self.sessions.clone();
        let cache_size = self.cache_size;
        let ttl = self.ttl;
        let engine = self.engine.clone();
        let vtb = self.visual_token_budget;
        let sys_for_task = system_prompt;
        let session_id_for_task = session_id.clone();
        let turns_for_task = turns;

        // All Conversation work — lookup, clone, send, re-cache —
        // happens on the blocking pool. Conversation::clone is a
        // tensor deep-copy and Conversation::send_message_stream
        // blocks until the C callback drains; both are unsafe to
        // run on the async runtime.
        let gen_task = smol::unblock(move || -> Result<String, String> {
            // ── Pick the warmest prefix ─────────────────────────
            //
            // Preference order:
            //   1. sessions[session_id] hash, if still cached, not
            //      TTL-expired, and the stored turn_count actually
            //      matches turns[..turn_count]'s hash (the verify
            //      guards against a stale session pin pointing at
            //      a hash that belongs to a different
            //      conversation).
            //   2. Longest-prefix scan from turns.len() down to 0
            //      (N == 0 == cold miss; first hash hit wins).
            //   3. Fresh Conversation built from the engine.
            let (mut conv, prefix_n) = {
                let mut cache = prefix_cache.lock().unwrap();

                let mut chosen: Option<(u64, usize)> = None;

                if let Some(h) = session_pin {
                    if let Some(c) = cache.get(&h) {
                        if c.last_used.elapsed() < ttl
                            && c.turn_count <= turns_for_task.len()
                        {
                            // Verify the pinned hash matches the
                            // canonical bytes of the incoming
                            // turns[..turn_count]. Cheap (one
                            // xxh3_128) and prevents nasty
                            // wrong-context KV reuse if a client
                            // reuses a session_id under a
                            // different conversation.
                            let verify = hash_prefix(
                                sys_for_task.as_deref(),
                                top_p_h,
                                temp_h,
                                &turns_for_task[..c.turn_count],
                            );
                            if verify == h {
                                chosen = Some((h, c.turn_count));
                            }
                        }
                    }
                }

                if chosen.is_none() {
                    for n in (0..=turns_for_task.len()).rev() {
                        let h = hash_prefix(
                            sys_for_task.as_deref(),
                            top_p_h,
                            temp_h,
                            &turns_for_task[..n],
                        );
                        if let Some(c) = cache.get(&h) {
                            if c.last_used.elapsed() < ttl {
                                chosen = Some((h, n));
                                break;
                            }
                        }
                    }
                }

                match chosen {
                    Some((h, n)) if n > 0 => {
                        // Clone the cached Conversation
                        // (KV-state deep copy via
                        // `litert_lm_conversation_clone`). The
                        // cache keeps the original donor; the
                        // clone is our working copy and is dropped
                        // at the end of this closure unless we
                        // re-cache it (which we do, after
                        // generation, under the new hash).
                        let entry =
                            cache.get_mut(&h).expect("entry just looked up");
                        entry.last_used = Instant::now();
                        let working = entry
                            .conv
                            .clone()
                            .map_err(|e| format!("litertlm Conversation::clone: {e}"))?;
                        log::info!(
                            "[litertlm] prefix-cache HIT n={n}/{} hash={h:016x}",
                            turns_for_task.len(),
                        );
                        (working, n)
                    }
                    _ => {
                        log::info!(
                            "[litertlm] prefix-cache MISS — fresh Conversation for {} turns",
                            turns_for_task.len(),
                        );
                        // Release the cache lock before the C call
                        // so a slow create_conversation doesn't
                        // block other lookups.
                        drop(cache);
                        let mut conv = engine
                            .create_conversation(sampler)
                            .map_err(|e| format!("litertlm create_conversation: {e}"))?;
                        if let Some(b) = vtb {
                            conv = conv.with_visual_token_budget(b);
                        }
                        (conv, 0)
                    }
                }
            };

            // ── Replay turns[prefix_n..] ─────────────────────────
            //
            // See the module-level docs for the cold-start replay
            // drift caveat. Only the FINAL turn's chunks go to the
            // wire; intermediate regenerated responses are
            // accumulated locally and discarded.
            let suffix = &turns_for_task[prefix_n..];
            let replay_msgs = sendable_contents(suffix);
            if replay_msgs.is_empty() {
                return Err(
                    "litertlm: nothing to send (suffix has no User/Tool turns)".into(),
                );
            }

            let mut last_response = String::new();
            for (i, msg) in replay_msgs.iter().enumerate() {
                let is_last = i + 1 == replay_msgs.len();
                let mut acc = String::new();
                conv.send_message_stream(msg, |chunk: &str| {
                    if is_last {
                        let _ = smol::block_on(tx.send(chunk.to_string()));
                    }
                    acc.push_str(chunk);
                })
                .map_err(|e| format!("litertlm send_message_stream: {e}"))?;
                if is_last {
                    last_response = acc;
                }
            }
            drop(tx);

            // ── Re-cache the advanced Conversation ──────────────
            //
            // The working `conv` now holds turns[..len] + the
            // freshly-generated assistant turn. We hash the full
            // logical conversation (original turns + new
            // assistant) — that's the prefix the next request will
            // look up. Clone `conv` once more into the cache so
            // the working copy itself stays scoped here (it falls
            // out at the end of the closure and the C side frees
            // its KV tensors). The cached clone is what future
            // requests will derive their working copies from.
            let mut full_turns = turns_for_task.clone();
            full_turns.push(Turn::assistant(last_response.clone()));
            let new_hash = hash_prefix(
                sys_for_task.as_deref(),
                top_p_h,
                temp_h,
                &full_turns,
            );
            let to_cache = conv
                .clone()
                .map_err(|e| format!("litertlm Conversation::clone (re-cache): {e}"))?;

            {
                let mut cache = prefix_cache.lock().unwrap();
                if cache.len() >= cache_size && !cache.contains_key(&new_hash) {
                    // LRU eviction by `last_used`. Re-inserting an
                    // existing key (same hash) doesn't grow the
                    // map, so skip eviction in that case.
                    if let Some(k) = cache
                        .iter()
                        .min_by_key(|(_, c)| c.last_used)
                        .map(|(k, _)| *k)
                    {
                        log::info!(
                            "[litertlm] prefix-cache evict {k:016x} (size={cache_size})",
                        );
                        cache.remove(&k);
                    }
                }
                cache.insert(
                    new_hash,
                    CachedConv {
                        conv: to_cache,
                        turn_count: full_turns.len(),
                        last_used: Instant::now(),
                    },
                );
            }

            if let Some(id) = session_id_for_task {
                sessions.lock().unwrap().insert(id, new_hash);
            }

            Ok(last_response)
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
