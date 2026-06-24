//! End-to-end test for `SyncClient` against a real `model_api_server`
//! using `StubSlot` (which echoes prompts back).
//!
//! Runs the server on a background thread + connects from the main
//! thread with the sync client. Verifies handshake exposes the
//! stub's model name, inference round-trips, and shutdown closes
//! cleanly.

use std::sync::Arc;
use std::time::{Duration, Instant};

use model_api_client::{SyncClient, sync::StreamEvent};
use model_api_proto::{
    GpuRole, InferenceConfig, InferenceInput, InferenceRequest, SessionMode,
    ToolDef, ToolMode, ToolParam, ToolResult,
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
        tools: Vec::new(),
        tool_results: Vec::new(),
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
fn tools_in_request_yield_tool_calls_in_response_via_stub() {
    // Round-trips the new tool-wire path: request carries a single
    // ToolDef + tool_mode=Client; StubSlot's tool-aware echo emits a
    // canned ToolCall; the response surfaces it back to the caller.
    // Doesn't exercise marker parsing (that's covered by
    // model_api_server::llama_slot::tests unit tests) — this is
    // the wire round-trip itself.
    let socket = ephemeral_socket("tools");
    spawn_server(&socket);
    wait_for_socket(&socket);

    let client = SyncClient::connect(&socket).expect("connect");

    let calc = ToolDef {
        name: "calculator".into(),
        description: "Evaluate a math expression".into(),
        parameters: vec![ToolParam {
            name: "expr".into(),
            param_type: "string".into(),
            required: true,
            description: "The expression to evaluate".into(),
        }],
    };

    let mut cfg = InferenceConfig::default();
    cfg.tool_mode = ToolMode::Client;

    let r = client.inference(InferenceRequest {
        role: GpuRole::Deep,
        input: InferenceInput::Text("what is 2+2?".into()),
        max_tokens: 64,
        session: SessionMode::Stateless,
        inference_config: cfg,
        tools: vec![calc],
        tool_results: Vec::new(),
        cache_hash: 0,
        stream: false,
        progress: false,
    }).expect("inference");

    assert_eq!(r.tool_calls.len(), 1, "expected 1 tool call, got {}", r.tool_calls.len());
    assert_eq!(r.tool_calls[0].name, "calculator");
    assert_eq!(r.tool_calls[0].id, "tc-0");
    // Stub emits empty args; pi-ai-style consumers would JSON.parse
    // this and inspect the schema.
    assert_eq!(r.tool_calls[0].arguments_json, "{}");
}

#[test]
fn tool_results_round_trip_through_request_field() {
    // Second-turn flow: client sends back the result of a prior
    // tool call. StubSlot doesn't actually consume them (it just
    // echoes), but the wire path must encode + decode them cleanly.
    let socket = ephemeral_socket("tres");
    spawn_server(&socket);
    wait_for_socket(&socket);

    let client = SyncClient::connect(&socket).expect("connect");

    let r = client.inference(InferenceRequest {
        role: GpuRole::Deep,
        input: InferenceInput::Text("here's the result".into()),
        max_tokens: 64,
        session: SessionMode::Persistent { session_id: "s-tres".into() },
        inference_config: InferenceConfig::default(),
        tools: Vec::new(),
        tool_results: vec![ToolResult {
            call_id: "tc-0".into(),
            output: "4".into(),
        }],
        cache_hash: 0,
        stream: false,
        progress: false,
    }).expect("inference");

    // Stub still echoes its input; tool_results don't affect its
    // output. Important is that the wire encode/decode worked.
    assert_eq!(r.text, "echo: here's the result");
    assert_eq!(r.session_id.as_deref(), Some("s-tres"));
    assert_eq!(r.tool_calls.len(), 0);
}

#[test]
fn inference_stream_receives_chunks_progress_and_final() {
    // Wire-side streaming test: StubSlot's run_stream emits a
    // canned sequence of Progress + Chunk events around its echoed
    // response. The test collects events into a buffer (the napi
    // bindings will do the same via ThreadsafeFunction) and asserts
    // ordering + completeness.
    let socket = ephemeral_socket("strm");
    spawn_server(&socket);
    wait_for_socket(&socket);

    let client = SyncClient::connect(&socket).expect("connect");

    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<StreamEvent>::new()));
    let ev = events.clone();
    let resp = client
        .inference_stream(req("hello streaming", "s-strm"), move |e| {
            ev.lock().unwrap().push(e);
        })
        .expect("inference_stream");

    assert_eq!(resp.text, "echo: hello streaming");

    let collected = events.lock().unwrap();
    // Expect at least: 2 progress (queued + gen_start) + 1+ chunk
    // + 1 progress (gen_done). StubSlot splits the text into ~3
    // chunks; exact count depends on length math but >=1 chunk is
    // the contract.
    let chunk_count = collected.iter().filter(|e| e.kind() == "chunk").count();
    let progress_count = collected.iter().filter(|e| e.kind() == "progress").count();
    assert!(
        chunk_count >= 1,
        "expected at least 1 chunk event, got {chunk_count} (events={collected:?})"
    );
    assert!(
        progress_count >= 2,
        "expected at least 2 progress events, got {progress_count} (events={collected:?})"
    );

    // First event should be a `queued` progress, last a `gen_done`.
    if let Some(StreamEvent::Progress(p)) = collected.first() {
        assert_eq!(p.phase, "queued");
    } else {
        panic!("first event should be Progress(queued), got {:?}", collected.first());
    }
    if let Some(StreamEvent::Progress(p)) = collected.last() {
        assert_eq!(p.phase, "gen_done");
    } else {
        panic!("last event should be Progress(gen_done), got {:?}", collected.last());
    }

    // Concatenated chunk deltas should reconstruct the full text.
    let reconstructed: String = collected.iter().filter_map(|e| match e {
        StreamEvent::Chunk(c) => Some(c.delta_text.clone()),
        _ => None,
    }).collect();
    assert_eq!(reconstructed, resp.text);
}

#[test]
fn cancel_sends_one_frame_and_returns_immediately() {
    // Smoke test that the wire frame goes out cleanly. The server
    // logs it but doesn't act today (cancellation is a TODO inside
    // the slot loop) — what we're verifying is the SyncClient
    // surface so napi bindings can wrap it for AbortSignal.
    let socket = ephemeral_socket("cancl");
    spawn_server(&socket);
    wait_for_socket(&socket);

    let client = SyncClient::connect(&socket).expect("connect");
    client.cancel().expect("cancel");
    // No assertion on server behaviour; if cancel ever returned an
    // error we'd want to know.
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
