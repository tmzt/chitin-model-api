//! Smoke client for chitin-model-api. Connects to the UDS socket,
//! sends one streaming inference, prints the response.
//!
//! Usage:
//!   smoke <socket> [--session <id>] "<prompt>"
//!
//! Without `--session`, the request is Stateless. With it, the
//! request is Persistent with the given session_id — run smoke
//! twice with the same id to exercise the LiteRtLmSlot
//! Conversation pool's multi-turn KV reuse.

use model_api_client::sync::{StreamEvent, SyncClient};
use model_api_proto::{
    GpuRole, InferenceConfig, InferenceInput, InferenceRequest, SessionMode,
};

fn usage() -> ! {
    eprintln!("usage: smoke <socket> [--session <id>] \"<prompt>\"");
    std::process::exit(2);
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let socket = args.next().unwrap_or_else(|| usage());

    let mut session_id: Option<String> = None;
    let mut prompt: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--session" => session_id = Some(args.next().unwrap_or_else(|| usage())),
            "-h" | "--help" => usage(),
            other => {
                if prompt.is_some() { usage(); }
                prompt = Some(other.to_string());
            }
        }
    }
    let prompt = prompt.unwrap_or_else(|| usage());

    let client = SyncClient::connect(&socket).map_err(|e| format!("connect: {e}"))?;
    let info = client.server_info();
    eprintln!(
        "[smoke] connected: model={} gpu_mb={:?} session={:?}",
        info.model_name, info.gpu_memory_mb, session_id.as_deref().unwrap_or("<stateless>"),
    );

    let session = match &session_id {
        Some(id) => SessionMode::Persistent { session_id: id.clone() },
        None => SessionMode::Stateless,
    };

    let t0 = std::time::Instant::now();
    let req = InferenceRequest {
        role: GpuRole::Deep,
        input: InferenceInput::Text(prompt),
        max_tokens: 128,
        session,
        inference_config: InferenceConfig::default(),
        tools: Vec::new(),
        tool_results: Vec::new(),
        cache_hash: 0,
        stream: true,
        progress: false,
    };
    let resp = client.inference_stream(req, |ev| match ev {
        StreamEvent::Chunk(c) => eprint!("{}", c.delta_text),
        StreamEvent::Progress(p) => eprintln!("[progress] {}", p.phase),
    }).map_err(|e| format!("inference_stream: {e}"))?;

    eprintln!();
    eprintln!("[smoke] elapsed={:?}", t0.elapsed());
    println!("\n--- final text ---\n{}", resp.text);
    Ok(())
}
