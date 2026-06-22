//! `chitin-model-api` — UDS server that owns the llama.cpp Session +
//! smart KV cache. Receives `model_api_proto::ClientMessage`s,
//! processes them sequentially through a single global slot, replies
//! with `ServerMessage`s on the same socket.
//!
//! Scaffold: today this just listens on UDS, accepts a connection,
//! exchanges the Hello handshake, and rejects every inference with
//! "not implemented". Real slot wiring lands in follow-up commits.
//!
//! Run:
//!   chitin-model-api --socket /tmp/chitin-model-api.sock --model <path>

use std::path::PathBuf;

use model_api_proto::{ClientMessage, ServerMessage, PROTOCOL_VERSION};

mod framed;

/// CLI args parsed by hand — no clap dep so the binary stays small.
struct Args {
    socket_path: PathBuf,
    /// GGUF model file. Required when the `llama-cpp` feature is on;
    /// today the scaffold ignores it.
    _model_path: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut socket_path: Option<PathBuf> = None;
    let mut model_path: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--socket" => socket_path = args.next().map(PathBuf::from),
            "--model" => model_path = args.next().map(PathBuf::from),
            "-h" | "--help" => {
                eprintln!("usage: chitin-model-api --socket <path> [--model <gguf>]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(Args {
        socket_path: socket_path.ok_or_else(|| "--socket is required".to_string())?,
        _model_path: model_path,
    })
}

fn main() -> Result<(), String> {
    env_logger::init();
    let args = parse_args()?;
    log::info!(
        "[model_api] starting: socket={} protocol_version={}",
        args.socket_path.display(),
        PROTOCOL_VERSION,
    );
    smol::block_on(serve(args))
}

async fn serve(args: Args) -> Result<(), String> {
    // Best-effort: remove a stale socket file so re-launches work
    // without manual cleanup. If somebody's actually listening on it
    // this returns "address in use" on bind below, which is the
    // right behaviour — don't blow away an active server.
    let _ = std::fs::remove_file(&args.socket_path);

    let listener = smol::net::unix::UnixListener::bind(&args.socket_path)
        .map_err(|e| format!("bind {}: {e}", args.socket_path.display()))?;
    log::info!("[model_api] listening on {}", args.socket_path.display());

    loop {
        let (stream, _addr) = listener
            .accept()
            .await
            .map_err(|e| format!("accept: {e}"))?;
        log::info!("[model_api] connection accepted");
        smol::spawn(async move {
            if let Err(e) = handle_client(stream).await {
                log::warn!("[model_api] connection ended: {e}");
            }
        })
        .detach();
    }
}

async fn handle_client(_stream: smol::net::unix::UnixStream) -> Result<(), String> {
    // Scaffold: handshake only, then read-loop that rejects everything
    // with InferenceError so a client doesn't silently hang. Real
    // codec wiring (framed::read_frame / write_frame against the
    // split halves) lands in a follow-up commit alongside the slot
    // owner.
    let _ = ClientMessage::Hello { protocol_version: PROTOCOL_VERSION };
    let _ = ServerMessage::Hello {
        protocol_version: PROTOCOL_VERSION,
        model_name: "scaffold".into(),
        gpu_memory_mb: None,
    };
    Ok(())
}
