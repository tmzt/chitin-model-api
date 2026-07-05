//! Prefix-cache persistence via `llama_state_seq_save_file` /
//! `llama_state_seq_load_file`. Lets a long-lived inference process
//! skip system-prompt prefill on a warm restart — the role
//! `thinker_impl::KvCacheMode::Static` already plays for the shady-
//! thinker path via its `EMBEDDED_PREFIX_CACHE`.
//!
//! Cache files land at `<model_dir>/llama_prefix_<tag>.bin`. The
//! format is llama.cpp's own per-sequence state dump — **not**
//! compatible with shady-thinker's prefix-cache files. First-run
//! rebuild on the llama_engine side is fine.
//!
//! Currently not wired into `lib::spawn` or `thinker_impl::llama_slot`
//! — call sites need a stable cache tag (typically a hash of the
//! system prompt), and the prefix prefill itself isn't separated
//! out yet. Both follow-ups land alongside the
//! `ThinkerConfig::kv_cache = KvCacheMode::Static(...)` plumbing
//! into the llama-cpp path.

use std::ffi::CString;
use std::path::Path;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::token::LlamaToken;

/// Returns the cache filename for a given model dir + tag.
pub fn cache_path(model_dir: &str, tag: &str) -> String {
    format!("{model_dir}/llama_prefix_{tag}.bin")
}

/// Save the current KV state of `seq_id` to `path`. `tokens` is the
/// token list that was prefilled into that sequence — written into
/// the cache file alongside the KV blob.
///
/// Returns the number of bytes written, or an error string.
pub fn save(
    ctx: &mut LlamaContext<'_>,
    path: &Path,
    seq_id: i32,
    tokens: &[LlamaToken],
) -> Result<usize, String> {
    let c_path = CString::new(path.to_string_lossy().as_bytes())
        .map_err(|e| format!("prefix_cache::save bad path: {e}"))?;
    // SAFETY: ctx pointer is valid for the duration of the call;
    // tokens is a borrowed slice; path is null-terminated.
    let written = unsafe {
        llama_cpp_sys_2::llama_state_seq_save_file(
            ctx_ptr_mut(ctx),
            c_path.as_ptr(),
            seq_id,
            tokens.as_ptr() as *const llama_cpp_sys_2::llama_token,
            tokens.len(),
        )
    };
    if written == 0 {
        Err(format!("llama_state_seq_save_file({}) returned 0", path.display()))
    } else {
        Ok(written)
    }
}

/// Load a previously-saved sequence into `dest_seq_id` of `ctx`. The
/// recovered token list is appended to `tokens_out` (caller pre-
/// allocates capacity = max prompt length).
///
/// Returns the number of bytes read, or an error string. Returns
/// `Ok(0)` cleanly if the file doesn't exist or is the wrong format —
/// callers fall back to fresh prefill.
pub fn load(
    ctx: &mut LlamaContext<'_>,
    path: &Path,
    dest_seq_id: i32,
    tokens_out: &mut Vec<LlamaToken>,
) -> Result<usize, String> {
    if !path.exists() {
        return Ok(0);
    }
    let c_path = CString::new(path.to_string_lossy().as_bytes())
        .map_err(|e| format!("prefix_cache::load bad path: {e}"))?;
    let capacity = tokens_out.capacity();
    if capacity == 0 {
        return Err("prefix_cache::load needs preallocated tokens_out capacity".into());
    }
    let mut n_tokens: usize = 0;
    let read = unsafe {
        llama_cpp_sys_2::llama_state_seq_load_file(
            ctx_ptr_mut(ctx),
            c_path.as_ptr(),
            dest_seq_id,
            tokens_out.as_mut_ptr() as *mut llama_cpp_sys_2::llama_token,
            capacity,
            &mut n_tokens as *mut usize,
        )
    };
    if read == 0 {
        // File present but couldn't be applied — wrong format, wrong
        // model, etc. Treat as cache miss, not a hard error.
        return Ok(0);
    }
    // SAFETY: llama.cpp wrote n_tokens token slots into the buffer.
    unsafe { tokens_out.set_len(n_tokens) };
    Ok(read)
}

/// Forward to `LlamaContext::as_raw_ptr_mut` (added on the
/// llama-cpp-2 fork). Mirrors the `LlamaModel::as_raw_ptr` accessor
/// the MTP scaffold uses.
fn ctx_ptr_mut(ctx: &mut LlamaContext<'_>) -> *mut llama_cpp_sys_2::llama_context {
    ctx.as_raw_ptr_mut()
}
