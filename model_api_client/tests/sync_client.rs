//! End-to-end test for `SyncClient` against a real `model_api_server`
//! using `StubSlot` (which echoes prompts back).
//!
//! Runs the server on a background thread + connects from the main
//! thread with the sync client. Verifies handshake exposes the
//! stub's model name, inference round-trips, and shutdown closes
//! cleanly.

use std::sync::Arc;
use std::time::{Duration, Instant};

use model_api_client::SyncClient;
use model_api_proto::{
    GpuRole, InferenceConfig, InferenceInput, InferenceRequest, SessionMode,
};
use model_api_server::{serve, slot::StubSlot};

fn ephemeral_socket(tag: &str) -> std::path::PathBuf {
    // Same SUN_LEN-respecting layout as the other tests.
    std::path::PathBuf::from(format!("/tmp/cm-sync-{}-{}.sock", std::process::id(), tag))
}

fn spawn_server(socket: &std::path::Path) {
    let s = socket.to_path_buf();
    std::thread::Builder::new()
        .name("model-api-sync-test-server".into())
        .spawn(move || {
            smol::block_on(async {
                let _ = serve(s, Arc::new(StubSlot::new("sync-test-stub"))).await;
            });
        })
        .unwrap();
}

fn wait_for_socket(socket: &std::path::Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if std::os::unix::net::UnixStream::connect(socket).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("server didn't start: {}", socket.display());
}

fn req(prompt: &str, session: &str) -> InferenceRequest {
    InferenceRequest {
        role: GpuRole::Deep,
        input: InferenceInput::Text(prompt.into()),
        max_tokens: 64,
        session: SessionMode::Persistent { session_id: session.into() },
        inference_config: InferenceConfig::default(),
        cache_hash: 0,
        stream: false,
        progress: false,
    }
}

#[test]
fn handshake_inference_shutdown_round_trip() {
    let socket = ephemeral_socket("rt");
    spawn_server(&socket);
    wait_for_socket(&socket);

    let client = SyncClient::connect(&socket).expect("connect");

    // Handshake info from the stub.
    let info = client.server_info();
    assert_eq!(info.model_name, "sync-test-stub");
    assert!(info.gpu_memory_mb.is_none());

    // Inference echoes the prompt.
    let resp = client.inference(req("hello sync", "s-1")).expect("inference");
    assert_eq!(resp.text, "echo: hello sync");
    assert_eq!(resp.session_id.as_deref(), Some("s-1"));

    // Shutdown drains a Goodbye.
    client.shutdown().expect("shutdown");
}

#[test]
fn two_sequential_requests_serialize_through_the_mutex() {
    let socket = ephemeral_socket("seq");
    spawn_server(&socket);
    wait_for_socket(&socket);

    let client = SyncClient::connect(&socket).expect("connect");
    let r1 = client.inference(req("first", "s-a")).expect("inf 1");
    let r2 = client.inference(req("second", "s-b")).expect("inf 2");
    assert_eq!(r1.text, "echo: first");
    assert_eq!(r2.text, "echo: second");
    assert_eq!(r1.session_id.as_deref(), Some("s-a"));
    assert_eq!(r2.session_id.as_deref(), Some("s-b"));
}
