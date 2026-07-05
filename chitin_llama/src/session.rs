//! `Session` owns the `LlamaBackend`, `LlamaModel`, and `LlamaContext`
//! and exposes both a one-shot `generate_raw` loop and the per-step
//! primitives (`prefill`, `step`, `push_token`) that `channels.rs`
//! drives. Single-threaded by design — instantiated and used inside
//! the inference thread spawned by `lib::spawn`.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

/// One loaded model + its inference context, plus per-call batch and
/// sampler state. `LlamaContext` borrows from `LlamaModel`; we extend
/// the model's lifetime via `transmute` so both live in one struct —
/// the legacy stub and llama-cpp-2's own server example use the same
/// trick. Safety: model and context have identical lifetimes (both
/// dropped in `Drop` order), and `Session` never crosses threads.
pub struct Session {
    _backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    ctx: LlamaContext<'static>,
    batch: LlamaBatch<'static>,
    /// `LlamaBatch` doesn't expose its allocated capacity; we mirror
    /// the value passed to `LlamaBatch::new` so `prefill` can chunk.
    batch_capacity: usize,
    sampler: LlamaSampler,
    n_cur: i32,
    max_seq_len: u32,
    /// Optional MTP / speculative-decode session. NULL when MTP isn't
    /// configured or when `llama_rs_mtp_init` returned NULL (stub).
    /// When non-null, `step()` proposes `mtp_draft_n` tokens via the
    /// draft model, verifies them against the target, and accepts the
    /// longest matching prefix.
    mtp: *mut llama_cpp_sys_2::llama_rs_mtp_session,
    mtp_draft_n: u32,
}

// SAFETY: Session holds a raw pointer into a llama.cpp MTP context
// which the underlying C++ code treats as single-owner. Callers are
// expected to hold the Session behind a Mutex (or otherwise guarantee
// exclusive access) so only one thread touches the FFI at a time.
// These impls exist so a Session can live inside an `Arc<Mutex<>>`
// dispatched from an async executor's blocking pool.
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

/// Build configuration for `Session::load`. Mirrors the GGUF-specific
/// subset of `lib::ThinkerConfig`.
pub struct LoadConfig {
    pub gguf_path: String,
    pub max_seq_len: u32,
    pub n_gpu_layers: i32,
    pub n_batch: u32,
    /// Optional draft model GGUF for MTP / speculative decode. `None`
    /// disables MTP; `Some` is honored only if `mtp_draft_n > 0`.
    /// Currently no-op even when set — see `wrapper_common.cpp`
    /// `llama_rs_mtp_init` stub.
    pub draft_gguf_path: Option<String>,
    /// Max draft tokens proposed per step. 0 disables MTP.
    pub mtp_draft_n: u32,
}

/// Process-wide `LlamaBackend`. `llama_backend_init()` may only be
/// called once per process — calling it a second time errors with
/// `BackendAlreadyInitialized`. Both the fast and deep thinker
/// threads load a `Session`, so we cache the first `init()` and
/// hand back clones to every later caller.
///
/// Also installs a log callback that routes llama.cpp + ggml output
/// through Rust's `log` crate, so GGML decode failures (which
/// otherwise print to stderr and are swallowed by `cargo run`) show
/// up in `chitin-ts-*.log`.
fn shared_backend() -> Result<Arc<LlamaBackend>, String> {
    use std::sync::{Mutex, OnceLock};
    static BACKEND: OnceLock<Mutex<Option<Arc<LlamaBackend>>>> = OnceLock::new();
    let slot = BACKEND.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().map_err(|e| format!("backend mutex: {e}"))?;
    if let Some(b) = guard.as_ref() {
        return Ok(b.clone());
    }
    let backend = Arc::new(
        LlamaBackend::init().map_err(|e| format!("llama backend init: {e}"))?,
    );
    install_log_callback();
    *guard = Some(backend.clone());
    Ok(backend)
}

/// Bridge llama.cpp / ggml C-side logs into Rust's `log` crate.
/// Without this they write to stderr and are lost when the process
/// is launched under `cargo run` with stdout/stderr captured.
unsafe extern "C" fn llama_log_bridge(
    level: llama_cpp_sys_2::ggml_log_level,
    text: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
    if text.is_null() {
        return;
    }
    let s = std::ffi::CStr::from_ptr(text).to_string_lossy();
    let trimmed = s.trim_end_matches('\n');
    if trimmed.is_empty() {
        return;
    }
    let rust_level = match level {
        llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR => log::Level::Error,
        llama_cpp_sys_2::GGML_LOG_LEVEL_WARN => log::Level::Warn,
        llama_cpp_sys_2::GGML_LOG_LEVEL_INFO => log::Level::Info,
        llama_cpp_sys_2::GGML_LOG_LEVEL_DEBUG => log::Level::Debug,
        _ => log::Level::Trace,
    };
    log::log!(target: "llama_cpp", rust_level, "{}", trimmed);
}

fn install_log_callback() {
    unsafe {
        llama_cpp_sys_2::llama_log_set(Some(llama_log_bridge), std::ptr::null_mut());
        llama_cpp_sys_2::ggml_log_set(Some(llama_log_bridge), std::ptr::null_mut());
    }
}

impl Session {
    /// Load a GGUF model and prepare a fresh inference context.
    pub fn load(cfg: LoadConfig) -> Result<Self, String> {
        let backend = shared_backend()?;

        log::info!(
            "llama_engine::session: loading {} (ctx={}, gpu_layers={})",
            cfg.gguf_path,
            cfg.max_seq_len,
            cfg.n_gpu_layers,
        );

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(cfg.n_gpu_layers as u32);

        let model = Arc::new(
            LlamaModel::load_from_file(&backend, Path::new(&cfg.gguf_path), &model_params)
                .map_err(|e| format!("load_from_file({}): {e}", cfg.gguf_path))?,
        );

        // `swa_full(false)` shrinks the sliding-window attention KV
        // cache to match `n_ctx` instead of allocating for the model's
        // full trained SWA window. For Gemma 4 (trained with a very
        // large SWA), the default (`true`) blows Apple Metal's working-
        // set budget the first time we decode anything, even with a
        // tiny prompt — see ggml log
        // `using full-size SWA cache` + `Insufficient Memory`. Honor
        // the caller's `max_seq_len` as the actual SWA bound.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(cfg.max_seq_len))
            .with_n_batch(cfg.n_batch)
            .with_swa_full(false);

        // SAFETY: see Session doc comment — model and context have
        // identical lifetimes and Session is single-threaded.
        let ctx = unsafe {
            let model_ref: &LlamaModel = &model;
            let model_static: &'static LlamaModel = std::mem::transmute(model_ref);
            model_static
                .new_context(&backend, ctx_params)
                .map_err(|e| format!("new_context: {e}"))?
        };

        let batch = LlamaBatch::new(cfg.n_batch as usize, 1);
        let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

        // Optional MTP speculative-decode session. Falls back to NULL
        // when `mtp_draft_n == 0`, when `llama_rs_mtp_init` returns
        // NULL (the current scaffold), or when the draft model can't
        // load. The single-token `step()` path is used in that case.
        let mtp = if cfg.mtp_draft_n > 0 {
            let draft_cstr = cfg
                .draft_gguf_path
                .as_ref()
                .map(|p| std::ffi::CString::new(p.as_str()).ok())
                .flatten();
            let draft_ptr = draft_cstr
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(std::ptr::null());
            // SAFETY: target_model pointer is borrowed from our Arc<LlamaModel>
            // and lives as long as the Session. The wrapper init returns
            // NULL (stub) or a heap-allocated session we own.
            unsafe {
                llama_cpp_sys_2::llama_rs_mtp_init(
                    model.as_raw_ptr(),
                    draft_ptr,
                    cfg.mtp_draft_n,
                )
            }
        } else {
            std::ptr::null_mut()
        };

        if cfg.mtp_draft_n > 0 && mtp.is_null() {
            log::info!(
                "llama_engine::session: MTP requested (n={}) but llama_rs_mtp_init returned NULL — falling back to single-token decode",
                cfg.mtp_draft_n,
            );
        }

        log::info!("llama_engine::session: ready (n_ctx={}, mtp={})", cfg.max_seq_len, !mtp.is_null());

        Ok(Self {
            _backend: backend,
            model,
            ctx,
            batch,
            batch_capacity: cfg.n_batch as usize,
            sampler,
            mtp,
            mtp_draft_n: cfg.mtp_draft_n,
            n_cur: 0,
            max_seq_len: cfg.max_seq_len,
        })
    }

    /// Clear the KV cache and reset per-request state. Call between
    /// unrelated requests when running without a prefix cache.
    /// Current KV cache write head (token position of the next prefill).
    /// Callers use this to decide when to compact / reset before the
    /// cache hits `max_seq_len`.
    pub fn n_cur(&self) -> i32 {
        self.n_cur
    }

    /// Maximum tokens this session's context can hold.
    pub fn max_seq_len(&self) -> u32 {
        self.max_seq_len
    }

    pub fn clear_kv(&mut self) {
        self.ctx.clear_kv_cache();
        self.n_cur = 0;
    }

    /// Save the current KV sequence (seq 0) to `path` via
    /// `llama_state_seq_save_file`. `tokens` is the token list that
    /// was prefilled into this sequence — written into the cache
    /// file header so `load_seq` can return it. Bumps `n_cur` is
    /// not modified (save is read-only on the KV state). Returns
    /// the number of bytes written.
    pub fn save_seq(
        &mut self,
        path: &std::path::Path,
        tokens: &[LlamaToken],
    ) -> Result<usize, String> {
        crate::prefix_cache::save(&mut self.ctx, path, 0, tokens)
    }

    /// Load a previously-saved sequence from `path` into seq 0,
    /// replacing the current KV. Returns the token list that was
    /// prefilled when the file was written (re-populated from the
    /// header) along with the byte count. `n_cur` is set to the
    /// number of tokens loaded so subsequent `prefill` extends
    /// from there. Returns `Ok((vec![], 0))` cleanly when the
    /// file is missing or unreadable — callers fall back to a
    /// fresh prefill.
    pub fn load_seq(
        &mut self,
        path: &std::path::Path,
        max_tokens: usize,
    ) -> Result<(Vec<LlamaToken>, usize), String> {
        let mut buf: Vec<LlamaToken> = Vec::with_capacity(max_tokens.max(1));
        let bytes = crate::prefix_cache::load(&mut self.ctx, path, 0, &mut buf)?;
        if bytes == 0 {
            return Ok((Vec::new(), 0));
        }
        self.n_cur = buf.len() as i32;
        Ok((buf, bytes))
    }

    /// Tokenize a prompt against the model's vocabulary.
    pub fn tokenize(&self, prompt: &str, add_bos: bool) -> Result<Vec<LlamaToken>, String> {
        let bos = if add_bos { AddBos::Always } else { AddBos::Never };
        self.model
            .str_to_token(prompt, bos)
            .map_err(|e| format!("tokenize: {e}"))
    }

    /// Decode a sequence of token ids to text, preserving special tokens
    /// (`<think>` / `<tool_call>` etc.) so the channel router can see them.
    pub fn detokenize(&self, tokens: &[LlamaToken]) -> String {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::with_capacity(tokens.len() * 4);
        for &tok in tokens {
            if let Ok(piece) = self.model.token_to_piece(tok, &mut decoder, true, None) {
                out.push_str(&piece);
            }
        }
        out
    }

    /// Decode a single token to text — used by the channel router to
    /// build up its raw_text buffer one token at a time.
    pub fn detokenize_one(&self, token: LlamaToken, decoder: &mut encoding_rs::Decoder) -> String {
        self.model
            .token_to_piece(token, decoder, true, None)
            .unwrap_or_default()
    }

    /// Is this the model's end-of-generation token?
    pub fn is_eog(&self, token: LlamaToken) -> bool {
        self.model.is_eog_token(token)
    }

    /// Prefill `prompt_tokens` into the KV cache STARTING AT
    /// `self.n_cur`. For a fresh / cleared KV that's position 0; for
    /// a continuation turn (caller did not call `clear_kv`) it's the
    /// running position from the previous turn, so the new tokens
    /// extend the existing sequence rather than colliding at 0
    /// (which would trip "inconsistent sequence positions").
    /// Returns the position after the prompt (the new `n_cur`).
    ///
    /// Chunks the prefill into `n_batch`-sized decodes so prompts
    /// longer than a single batch (typical for chat with system +
    /// history) don't overflow `LlamaBatch::add`'s "Insufficient
    /// Space" cap. Only the FINAL token of the FINAL chunk gets
    /// logits requested — `step()` samples from there.
    pub fn prefill(&mut self, prompt_tokens: &[LlamaToken]) -> Result<i32, String> {
        if prompt_tokens.is_empty() {
            return Err("empty prompt".into());
        }
        let start_pos = self.n_cur as usize;
        let end_total = start_pos + prompt_tokens.len();
        if end_total >= self.max_seq_len as usize {
            return Err(format!(
                "prompt {} tokens at pos {} exceeds context {}",
                prompt_tokens.len(),
                start_pos,
                self.max_seq_len,
            ));
        }

        let total = prompt_tokens.len();
        let chunk = self.batch_capacity;
        let mut i = 0;
        while i < total {
            let end = (i + chunk).min(total);
            self.batch.clear();
            for (j, &tok) in prompt_tokens[i..end].iter().enumerate() {
                let pos = (start_pos + i + j) as i32;
                let is_last = end == total && (i + j) == total - 1;
                self.batch
                    .add(tok, pos, &[0], is_last)
                    .map_err(|e| format!("batch add prefill: {e}"))?;
            }
            self.ctx
                .decode(&mut self.batch)
                .map_err(|e| format!("decode prefill: {e}"))?;
            i = end;
        }

        self.n_cur = end_total as i32;
        Ok(self.n_cur)
    }

    /// Sample the next token from the current logits. Returns `None`
    /// when the model emits its end-of-generation token.
    pub fn step(&mut self) -> Option<LlamaToken> {
        let token = self.sampler.sample(&self.ctx, self.batch.n_tokens() - 1);
        self.sampler.accept(token);
        if self.model.is_eog_token(token) {
            None
        } else {
            Some(token)
        }
    }

    /// Advance the KV cache by one token. Used by the channel router
    /// when injecting tool responses and for normal continuation.
    pub fn push_token(&mut self, token: LlamaToken) -> Result<(), String> {
        self.batch.clear();
        self.batch
            .add(token, self.n_cur, &[0], true)
            .map_err(|e| format!("batch add: {e}"))?;
        self.n_cur += 1;
        self.ctx
            .decode(&mut self.batch)
            .map_err(|e| format!("decode step: {e}"))
    }

    /// Returns true when MTP / speculative-decode is active for this
    /// session. Useful for callers (channels.rs, lib::spawn) that want
    /// to log or branch on the fast path.
    pub fn mtp_enabled(&self) -> bool {
        !self.mtp.is_null()
    }

    /// Borrow the loaded model. Used by dispatchers that need to
    /// tokenize / detokenize text against the GGUF's built-in vocab
    /// without taking a full `&Session` (which would conflict with
    /// the mutable borrow `generate_with_channels` takes).
    pub fn model(&self) -> &Arc<LlamaModel> {
        &self.model
    }

    /// Number of draft tokens proposed per step when MTP is active.
    pub fn mtp_draft_n(&self) -> u32 {
        self.mtp_draft_n
    }

    /// Run the generation loop with no channel routing — produced
    /// tokens are returned as a flat sequence. Used by callers that
    /// don't care about Gemma's `<think>` / `<tool_call>` markers
    /// (e.g. structured-output workers that only consume the final
    /// text).
    pub fn generate_raw(
        &mut self,
        prompt_tokens: &[LlamaToken],
        max_tokens: usize,
    ) -> Result<Vec<LlamaToken>, String> {
        self.prefill(prompt_tokens)?;
        let mut produced = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let Some(token) = self.step() else { break };
            produced.push(token);
            if self.push_token(token).is_err() {
                break;
            }
        }
        Ok(produced)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.mtp.is_null() {
            // SAFETY: pointer came from llama_rs_mtp_init; we own it.
            unsafe { llama_cpp_sys_2::llama_rs_mtp_free(self.mtp) };
            self.mtp = std::ptr::null_mut();
        }
    }
}
