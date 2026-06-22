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
    #[allow(dead_code)] // wired in stage 2's llama-cpp impl
    model_path: Option<PathBuf>,
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
        model_path,
    })
}

fn main() -> Result<(), String> {
    env_logger::init();
    let args = parse_args()?;

    // Pick the production slot. `llama-cpp` feature gates the real
    // wiring; without it we fall back to the stub so the binary at
    // least serves a connection and reports "no backend" cleanly.
    let slot: Arc<dyn SlotHandle> = {
        #[cfg(feature = "llama-cpp")]
        {
            // Real llama-cpp wiring lands in a follow-up commit. For
            // now we use the stub so the binary builds with or
            // without the feature.
            let _ = args.model_path; // silence dead-code under stub
            Arc::new(model_api_server::slot::StubSlot::new("stub-llama"))
        }
        #[cfg(not(feature = "llama-cpp"))]
        {
            Arc::new(model_api_server::slot::StubSlot::new("stub-no-backend"))
        }
    };

    smol::block_on(serve(args.socket_path, slot))
}
