//! `chitin-model-api` — UDS server that owns the llama.cpp Session +
//! smart KV cache. Thin shim: parses CLI, builds the production
//! slot, hands off to `model_api_server::serve`.
//!
//! Run:
//!   chitin-model-api --socket /tmp/chitin-model-api.sock --model <path>

use std::path::PathBuf;
use std::sync::Arc;

use model_api_server::{serve, SlotHandle};

struct Args {
    socket_path: PathBuf,
    #[cfg_attr(not(feature = "llama-cpp"), allow(dead_code))]
    model_path: Option<PathBuf>,
    /// Optional override for the model's display name reported in
    /// the Hello reply. Defaults to the GGUF file's stem.
    #[cfg_attr(not(feature = "llama-cpp"), allow(dead_code))]
    model_name: Option<String>,
    /// Cap on tokens per inference. Server-wide today; per-request
    /// override would come over the wire via `InferenceRequest::max_tokens`.
    #[cfg_attr(not(feature = "llama-cpp"), allow(dead_code))]
    max_tokens: usize,
    /// KV cache size. Bigger = longer context but more GPU memory.
    #[cfg_attr(not(feature = "llama-cpp"), allow(dead_code))]
    max_seq_len: u32,
}

fn parse_args() -> Result<Args, String> {
    let mut socket_path: Option<PathBuf> = None;
    let mut model_path: Option<PathBuf> = None;
    let mut model_name: Option<String> = None;
    let mut max_tokens: usize = 4096;
    let mut max_seq_len: u32 = 16384;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--socket" => socket_path = args.next().map(PathBuf::from),
            "--model" => model_path = args.next().map(PathBuf::from),
            "--model-name" => model_name = args.next(),
            "--max-tokens" => {
                max_tokens = args.next()
                    .ok_or_else(|| "--max-tokens needs a value".to_string())?
                    .parse().map_err(|e| format!("--max-tokens: {e}"))?;
            }
            "--max-seq-len" => {
                max_seq_len = args.next()
                    .ok_or_else(|| "--max-seq-len needs a value".to_string())?
                    .parse().map_err(|e| format!("--max-seq-len: {e}"))?;
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: chitin-model-api --socket <path> \\
                          [--model <gguf>] [--model-name <name>] \\
                          [--max-tokens <N>] [--max-seq-len <N>]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(Args {
        socket_path: socket_path.ok_or_else(|| "--socket is required".to_string())?,
        model_path,
        model_name,
        max_tokens,
        max_seq_len,
    })
}

fn main() -> Result<(), String> {
    env_logger::init();
    let args = parse_args()?;

    let slot: Arc<dyn SlotHandle> = build_slot(&args)?;

    smol::block_on(serve(args.socket_path, slot))
}

#[cfg(feature = "llama-cpp")]
fn build_slot(args: &Args) -> Result<Arc<dyn SlotHandle>, String> {
    let model_path = args.model_path.as_ref()
        .ok_or_else(|| "--model <gguf> is required with the llama-cpp feature".to_string())?;
    let model_dir = model_path.parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let gguf_path = Some(model_path.to_string_lossy().to_string());
    let model_name = args.model_name.clone().unwrap_or_else(|| {
        model_path.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    });
    log::info!(
        "[model_api] loading model: {} (name={}, max_tokens={}, max_seq_len={})",
        model_path.display(), model_name, args.max_tokens, args.max_seq_len,
    );
    let slot = model_api_server::llama_slot::LlamaSlot::spawn(
        model_dir, gguf_path, model_name, args.max_tokens, args.max_seq_len,
    )?;
    Ok(Arc::new(slot))
}

#[cfg(not(feature = "llama-cpp"))]
fn build_slot(_args: &Args) -> Result<Arc<dyn SlotHandle>, String> {
    log::warn!(
        "[model_api] built without `llama-cpp` feature — \
         serving with StubSlot (echoes prompts). Rebuild with \
         --features llama-cpp for the real backend."
    );
    Ok(Arc::new(model_api_server::slot::StubSlot::new("stub-no-backend")))
}
