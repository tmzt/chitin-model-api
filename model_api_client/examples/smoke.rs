//! Smoke client for chitin-model-api. Connects to the UDS socket,
//! sends one InferenceRequest, prints the response.
//!
//!   chitin-model-api-smoke <socket> "<prompt>"

use model_api_client::sync::{StreamEvent, SyncClient};
use model_api_proto::{
    GpuRole, InferenceConfig, InferenceInput, InferenceRequest, SessionMode,
};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let socket = args.next().ok_or("usage: smoke <socket> <prompt>")?;
    let prompt = args.next().ok_or("usage: smoke <socket> <prompt>")?;

    let client = SyncClient::connect(&socket).map_err(|e| format!("connect: {e}"))?;
    let info = client.server_info();
    eprintln!("[smoke] connected: model={} gpu_mb={:?}", info.model_name, info.gpu_memory_mb);

    let t0 = std::time::Instant::now();
    let req = InferenceRequest {
        role: GpuRole::Deep,
        input: InferenceInput::Text(prompt),
        max_tokens: 128,
        session: SessionMode::Stateless,
        inference_config: InferenceConfig::default(),
        tools: Vec::new(),
        tool_results: Vec::new(),
        cache_hash: 0,
        stream: true,
        progress: false,
    };
    let resp = client.inference_stream(
        req,
        |ev| match ev {
            StreamEvent::Chunk(c) => eprint!("{}", c.delta_text),
            StreamEvent::Progress(p) => eprintln!("[progress] {}", p.phase),
        },
    ).map_err(|e| format!("inference_stream: {e}"))?;

    eprintln!();
    let dt = t0.elapsed();
    eprintln!("[smoke] elapsed={:?}", dt);
    println!("\n--- final text ---\n{}", resp.text);
    Ok(())
}
