//! `chitin-model-api` — UDS server that owns the llama.cpp Session +
//! smart KV cache. Thin shim: parses CLI, builds the production
//! slot, hands off to `model_api_server::serve`.
//!
//! Run:
//!   chitin-model-api --socket /tmp/chitin-model-api.sock --model <path>

use std::path::PathBuf;
use std::sync::Arc;

use model_api_server::{serve, SlotHandle};

/// Which backend the server should drive at startup. Determined by
/// the `--backend` flag (with sensible defaults based on which
/// features are compiled in). Each variant maps to a [`build_slot`]
/// arm.
#[derive(Debug, Clone, Copy)]
enum BackendKind {
    /// In-process llama.cpp via thinker_impl (the `llama-cpp` feature).
    LlamaCpp,
    /// In-process LiteRT-LM via the `litertlm` crate (the
    /// `litert-lm` feature). Pixel demo path — PowerVR OpenCL.
    LiteRtLm,
    /// Subprocess wrapping `llama-completion` from llama.cpp. No
    /// in-process linking required — works regardless of features.
    Subprocess,
    /// StubSlot — echoes prompts. Test/integration default.
    Stub,
}

impl BackendKind {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "llama-cpp" | "llama" => Ok(Self::LlamaCpp),
            "litert-lm" | "litertlm" => Ok(Self::LiteRtLm),
            "subprocess" => Ok(Self::Subprocess),
            "stub" => Ok(Self::Stub),
            other => Err(format!("unknown --backend: {other}")),
        }
    }
}

struct Args {
    socket_path: PathBuf,
    /// Backend selection. When unset, defaults to the first
    /// compiled-in real backend, falling back to Subprocess (if
    /// --llama-bin/--model are present) and then Stub.
    backend: Option<BackendKind>,
    #[cfg_attr(not(any(feature = "llama-cpp", feature = "litert-lm")), allow(dead_code))]
    model_path: Option<PathBuf>,
    /// Optional override for the model's display name reported in
    /// the Hello reply. Defaults to the model file's stem.
    #[cfg_attr(not(any(feature = "llama-cpp", feature = "litert-lm")), allow(dead_code))]
    model_name: Option<String>,
    /// Cap on tokens per inference. Server-wide today; per-request
    /// override would come over the wire via `InferenceRequest::max_tokens`.
    #[cfg_attr(not(feature = "llama-cpp"), allow(dead_code))]
    max_tokens: usize,
    /// KV cache size. Bigger = longer context but more GPU memory.
    #[cfg_attr(not(feature = "llama-cpp"), allow(dead_code))]
    max_seq_len: u32,
    /// Path to llama-completion (or compatible) binary for the
    /// subprocess slot. When set + `--model` set, the server runs
    /// with a `SubprocessSlot` instead of `StubSlot` (no in-process
    /// llama-cpp linking required).
    llama_bin: Option<PathBuf>,
    /// LD_LIBRARY_PATH for the subprocess child. Defaults to
    /// `--llama-bin`'s parent dir.
    llama_lib_dir: Option<PathBuf>,
    /// Layers to offload to GPU for subprocess-slot inference.
    /// 0 = CPU only (default on Pixel — PowerVR Vulkan Q5/Q8
    /// shaders don't compile yet). 99 = all layers.
    llama_ngl: i32,
    /// LiteRT-LM accelerator backend. "gpu" (default) targets the
    /// device's GPU via the litertlm crate's GPU delegate (OpenCL
    /// on Android — what we want on Pixel/PowerVR). "cpu" falls
    /// back to the LiteRT CPU reference kernels.
    #[cfg_attr(not(feature = "litert-lm"), allow(dead_code))]
    litertlm_accel: String,
    /// Cap on tokens for a single LiteRT-LM Engine instance. Mirrors
    /// litert_lm_main's --max_num_tokens; the model file may override.
    #[cfg_attr(not(feature = "litert-lm"), allow(dead_code))]
    litertlm_max_num_tokens: i32,
    /// Visual token budget for multimodal LiteRT-LM models. Required
    /// for Gemma-4 E2B / E4B / 12B (text-only sends still need a
    /// positive value or the C side rejects with INVALID_ARGUMENT).
    /// 0 -> pass NULL (text-only model files).
    #[cfg_attr(not(feature = "litert-lm"), allow(dead_code))]
    litertlm_visual_token_budget: i32,
    /// LiteRT-LM Conversation pool capacity (LRU). Default 8.
    /// Multi-turn voice conversations keep one Conversation per
    /// distinct `session_id`; the pool serves N concurrent
    /// sessions before evicting the least-recently-used.
    #[cfg_attr(not(feature = "litert-lm"), allow(dead_code))]
    litertlm_session_pool_size: usize,
    /// TTL after which a pooled Conversation is treated as a cold
    /// miss (re-created from scratch). Default 900 seconds.
    /// Prevents stale sessions from holding GPU memory after the
    /// user has clearly moved on.
    #[cfg_attr(not(feature = "litert-lm"), allow(dead_code))]
    litertlm_session_ttl_secs: u64,
}

fn parse_args() -> Result<Args, String> {
    let mut socket_path: Option<PathBuf> = None;
    let mut backend: Option<BackendKind> = None;
    let mut model_path: Option<PathBuf> = None;
    let mut model_name: Option<String> = None;
    let mut max_tokens: usize = 4096;
    let mut max_seq_len: u32 = 16384;
    let mut llama_bin: Option<PathBuf> = None;
    let mut llama_lib_dir: Option<PathBuf> = None;
    let mut llama_ngl: i32 = 0;
    let mut litertlm_accel: String = "gpu".into();
    let mut litertlm_max_num_tokens: i32 = 4096;
    let mut litertlm_visual_token_budget: i32 = 512;
    let mut litertlm_session_pool_size: usize = 8;
    let mut litertlm_session_ttl_secs: u64 = 900;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--socket" => socket_path = args.next().map(PathBuf::from),
            "--backend" => {
                let v = args.next().ok_or_else(|| "--backend needs a value".to_string())?;
                backend = Some(BackendKind::parse(&v)?);
            }
            "--model" => model_path = args.next().map(PathBuf::from),
            "--model-name" => model_name = args.next(),
            "--llama-bin" => llama_bin = args.next().map(PathBuf::from),
            "--llama-lib-dir" => llama_lib_dir = args.next().map(PathBuf::from),
            "--llama-ngl" => {
                llama_ngl = args.next()
                    .ok_or_else(|| "--llama-ngl needs a value".to_string())?
                    .parse().map_err(|e| format!("--llama-ngl: {e}"))?;
            }
            "--litertlm-accel" => {
                litertlm_accel = args.next()
                    .ok_or_else(|| "--litertlm-accel needs a value".to_string())?;
            }
            "--litertlm-max-num-tokens" => {
                litertlm_max_num_tokens = args.next()
                    .ok_or_else(|| "--litertlm-max-num-tokens needs a value".to_string())?
                    .parse().map_err(|e| format!("--litertlm-max-num-tokens: {e}"))?;
            }
            "--litertlm-visual-token-budget" => {
                litertlm_visual_token_budget = args.next()
                    .ok_or_else(|| "--litertlm-visual-token-budget needs a value".to_string())?
                    .parse().map_err(|e| format!("--litertlm-visual-token-budget: {e}"))?;
            }
            "--litertlm-session-pool-size" => {
                litertlm_session_pool_size = args.next()
                    .ok_or_else(|| "--litertlm-session-pool-size needs a value".to_string())?
                    .parse().map_err(|e| format!("--litertlm-session-pool-size: {e}"))?;
            }
            "--litertlm-session-ttl-secs" => {
                litertlm_session_ttl_secs = args.next()
                    .ok_or_else(|| "--litertlm-session-ttl-secs needs a value".to_string())?
                    .parse().map_err(|e| format!("--litertlm-session-ttl-secs: {e}"))?;
            }
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
                          [--backend llama-cpp|litert-lm|subprocess|stub] \\
                          [--model <file>] [--model-name <name>] \\
                          [--llama-bin <path>] [--llama-lib-dir <path>] \\
                          [--llama-ngl <N>] \\
                          [--litertlm-accel gpu|cpu] [--litertlm-max-num-tokens <N>] \\
                          [--litertlm-visual-token-budget <N>] \\
                          [--litertlm-session-pool-size <N>] \\
                          [--litertlm-session-ttl-secs <S>] \\
                          [--max-tokens <N>] [--max-seq-len <N>]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(Args {
        socket_path: socket_path.ok_or_else(|| "--socket is required".to_string())?,
        backend,
        model_path,
        model_name,
        max_tokens,
        max_seq_len,
        llama_bin,
        llama_lib_dir,
        llama_ngl,
        litertlm_accel,
        litertlm_max_num_tokens,
        litertlm_visual_token_budget,
        litertlm_session_pool_size,
        litertlm_session_ttl_secs,
    })
}

fn main() -> Result<(), String> {
    env_logger::init();
    let args = parse_args()?;

    let kind = args.backend.unwrap_or_else(|| default_backend(&args));
    log::info!("[model_api] backend: {kind:?}");
    let slot: Arc<dyn SlotHandle> = build_slot(kind, &args)?;

    smol::block_on(serve(args.socket_path, slot))
}

/// When the user doesn't pass `--backend`, pick the first viable
/// backend: in-process litert-lm or llama-cpp if compiled in,
/// otherwise subprocess (when its CLI deps are present), else stub.
fn default_backend(args: &Args) -> BackendKind {
    #[cfg(feature = "litert-lm")]
    {
        if args.model_path.is_some() { return BackendKind::LiteRtLm; }
    }
    #[cfg(feature = "llama-cpp")]
    {
        if args.model_path.is_some() { return BackendKind::LlamaCpp; }
    }
    if args.llama_bin.is_some() && args.model_path.is_some() {
        return BackendKind::Subprocess;
    }
    BackendKind::Stub
}

fn build_slot(kind: BackendKind, args: &Args) -> Result<Arc<dyn SlotHandle>, String> {
    match kind {
        BackendKind::LlamaCpp => build_llama_slot(args),
        BackendKind::LiteRtLm => build_litertlm_slot(args),
        BackendKind::Subprocess => build_subprocess_slot(args),
        BackendKind::Stub => {
            log::warn!("[model_api] serving with StubSlot (echoes prompts)");
            Ok(Arc::new(model_api_server::slot::StubSlot::new("stub-no-backend")))
        }
    }
}

#[cfg(feature = "llama-cpp")]
fn build_llama_slot(args: &Args) -> Result<Arc<dyn SlotHandle>, String> {
    let model_path = args.model_path.as_ref()
        .ok_or_else(|| "--model <gguf> is required for --backend llama-cpp".to_string())?;
    let model_dir = model_path.parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let gguf_path = Some(model_path.to_string_lossy().to_string());
    let model_name = args.model_name.clone().unwrap_or_else(|| derive_model_name(model_path));
    log::info!(
        "[model_api] llama-cpp loading: {} (name={}, max_tokens={}, max_seq_len={})",
        model_path.display(), model_name, args.max_tokens, args.max_seq_len,
    );
    let slot = model_api_server::llama_slot::LlamaSlot::spawn(
        model_dir, gguf_path, model_name, args.max_tokens, args.max_seq_len,
    )?;
    Ok(Arc::new(slot))
}

#[cfg(not(feature = "llama-cpp"))]
fn build_llama_slot(_args: &Args) -> Result<Arc<dyn SlotHandle>, String> {
    Err("--backend llama-cpp requires building with --features llama-cpp".into())
}

#[cfg(feature = "litert-lm")]
fn build_litertlm_slot(args: &Args) -> Result<Arc<dyn SlotHandle>, String> {
    let model_path = args.model_path.as_ref()
        .ok_or_else(|| "--model <litertlm> is required for --backend litert-lm".to_string())?;
    let model_name = args.model_name.clone().unwrap_or_else(|| derive_model_name(model_path));
    let backend = match args.litertlm_accel.as_str() {
        "gpu" => litertlm::Backend::Gpu,
        "cpu" => litertlm::Backend::Cpu,
        other => return Err(format!("--litertlm-accel: expected 'gpu' or 'cpu', got '{other}'")),
    };
    let vtb = if args.litertlm_visual_token_budget > 0 {
        Some(args.litertlm_visual_token_budget)
    } else {
        None
    };
    let ttl = std::time::Duration::from_secs(args.litertlm_session_ttl_secs);
    log::info!(
        "[model_api] litert-lm loading: {} (name={}, accel={}, max_num_tokens={}, \
         vtb={:?}, pool_size={}, ttl={:?})",
        model_path.display(), model_name, args.litertlm_accel, args.litertlm_max_num_tokens,
        vtb, args.litertlm_session_pool_size, ttl,
    );
    let slot = model_api_server::litertlm_slot::LiteRtLmSlot::new(
        model_path.clone(),
        model_name,
        backend,
        args.litertlm_max_num_tokens,
        vtb,
        args.litertlm_session_pool_size,
        ttl,
    )?;
    Ok(Arc::new(slot))
}

#[cfg(not(feature = "litert-lm"))]
fn build_litertlm_slot(_args: &Args) -> Result<Arc<dyn SlotHandle>, String> {
    Err("--backend litert-lm requires building with --features litert-lm".into())
}

fn build_subprocess_slot(args: &Args) -> Result<Arc<dyn SlotHandle>, String> {
    let bin = args.llama_bin.as_ref()
        .ok_or_else(|| "--backend subprocess requires --llama-bin".to_string())?;
    let model = args.model_path.as_ref()
        .ok_or_else(|| "--backend subprocess requires --model".to_string())?;
    let lib_dir = args
        .llama_lib_dir
        .clone()
        .or_else(|| bin.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    log::info!(
        "[model_api] subprocess: bin={} model={} lib_dir={} ngl={}",
        bin.display(), model.display(), lib_dir.display(), args.llama_ngl,
    );
    Ok(Arc::new(model_api_server::subprocess_slot::SubprocessSlot::new(
        bin.clone(), model.clone(), lib_dir, args.llama_ngl,
    )))
}

#[cfg(any(feature = "llama-cpp", feature = "litert-lm"))]
fn derive_model_name(p: &PathBuf) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
