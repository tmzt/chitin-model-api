//! End-to-end server test: bind the server on an ephemeral UDS,
//! connect a raw client using the same framed codec, walk through
//! Hello → Inference → Shutdown.
//!
//! Uses [`model_api_server::slot::StubSlot`] so the test doesn't
//! need a GGUF model. The bridge code we're exercising is the same
//! the production binary uses.

use std::sync::Arc;

use model_api_proto::{
    ClientMessage, GpuRole, InferenceConfig, InferenceInput, InferenceRequest,
    ServerMessage, SessionMode, PROTOCOL_VERSION,
};
use model_api_server::{framed, serve, slot::StubSlot, SlotHandle};

/// Pick an ephemeral socket path keyed on pid + tag. Lives under
/// `/tmp` rather than `std::env::temp_dir()` because the macOS
/// per-user temp dir blows past SUN_LEN (104 chars) on its own.
fn ephemeral_socket(tag: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    std::path::PathBuf::from(format!("/tmp/cm-srv-{pid}-{tag}.sock"))
}

/// Start `serve` in the background on the given socket, return a
/// guard that removes the socket on drop. The server task runs
/// forever in normal use; we don't bother killing it explicitly —
/// the test process exits, taking the smol executor with it.
fn spawn_server(socket: &std::path::Path, slot: Arc<dyn SlotHandle>) {
    let s = socket.to_path_buf();
    std::thread::Builder::new()
        .name("model-api-server-test".into())
        .spawn(move || {
            smol::block_on(async {
                if let Err(e) = serve(s, slot).await {
                    eprintln!("[test-server] exited: {e}");
                }
            });
        })
        .unwrap();
}

/// Poll-connect: the server is started on another thread; give it a
/// few hundred ms of retries to bind before we connect.
fn connect_with_retry(socket: &std::path::Path) -> smol::net::unix::UnixStream {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    loop {
        match smol::block_on(smol::net::unix::UnixStream::connect(socket)) {
            Ok(s) => return s,
            Err(_) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(e) => panic!("connect {}: {e}", socket.display()),
        }
    }
}

#[test]
fn handshake_then_echo_inference() {
    let socket = ephemeral_socket("echo");
    let slot = Arc::new(StubSlot::new("test-stub-1"));
    spawn_server(&socket, slot);

    let stream = connect_with_retry(&socket);
    smol::block_on(async move {
        let (mut r, mut w) = futures_lite::io::split(stream);

        // Handshake.
        framed::write_frame_async(
            &mut w,
            &ClientMessage::Hello { protocol_version: PROTOCOL_VERSION },
        ).await.unwrap();

        let hello: ServerMessage = framed::read_frame_async(&mut r).await.unwrap().unwrap();
        match hello {
            ServerMessage::Hello { protocol_version, model_name, gpu_memory_mb } => {
                assert_eq!(protocol_version, PROTOCOL_VERSION);
                assert_eq!(model_name, "test-stub-1");
                assert!(gpu_memory_mb.is_none());
            }
            other => panic!("expected Hello, got {other:?}"),
        }

        // Inference — stub echoes the prompt back.
        framed::write_frame_async(&mut w, &ClientMessage::Inference(InferenceRequest {
            role: GpuRole::Deep,
            input: InferenceInput::Text("ping".into()),
            max_tokens: 64,
            session: SessionMode::Persistent { session_id: "s-7".into() },
            inference_config: InferenceConfig::default(),
            tools: Vec::new(),
            tool_results: Vec::new(),
            cache_hash: 0,
            stream: false,
            progress: false,
        })).await.unwrap();

        let reply: ServerMessage = framed::read_frame_async(&mut r).await.unwrap().unwrap();
        match reply {
            ServerMessage::InferenceComplete(resp) => {
                assert_eq!(resp.text, "echo: ping");
                assert_eq!(resp.session_id.as_deref(), Some("s-7"));
                assert_eq!(resp.raw_text.as_deref(), Some("<raw>echo: ping</raw>"));
            }
            other => panic!("expected InferenceComplete, got {other:?}"),
        }

        // Shutdown — server replies with Goodbye and closes.
        framed::write_frame_async(&mut w, &ClientMessage::Shutdown).await.unwrap();
        let bye: ServerMessage = framed::read_frame_async(&mut r).await.unwrap().unwrap();
        assert!(matches!(bye, ServerMessage::Goodbye));

        let eof: Option<ServerMessage> = framed::read_frame_async(&mut r).await.unwrap();
        assert!(eof.is_none());
    });
}

#[test]
fn protocol_version_mismatch_is_rejected() {
    let socket = ephemeral_socket("vermis");
    let slot = Arc::new(StubSlot::new("test-stub-2"));
    spawn_server(&socket, slot);

    let stream = connect_with_retry(&socket);
    smol::block_on(async move {
        let (mut r, mut w) = futures_lite::io::split(stream);
        framed::write_frame_async(
            &mut w,
            &ClientMessage::Hello { protocol_version: u32::MAX },
        ).await.unwrap();

        // Server sends an InferenceError then drops.
        let msg: ServerMessage = framed::read_frame_async(&mut r).await.unwrap().unwrap();
        match msg {
            ServerMessage::InferenceError { message } => {
                assert!(message.contains("protocol mismatch"), "got: {message}");
            }
            other => panic!("expected InferenceError, got {other:?}"),
        }
    });
}

#[test]
fn two_sequential_inferences_share_one_connection() {
    let socket = ephemeral_socket("twoinf");
    let slot = Arc::new(StubSlot::new("test-stub-3"));
    spawn_server(&socket, slot);

    let stream = connect_with_retry(&socket);
    smol::block_on(async move {
        let (mut r, mut w) = futures_lite::io::split(stream);

        // Handshake.
        framed::write_frame_async(
            &mut w,
            &ClientMessage::Hello { protocol_version: PROTOCOL_VERSION },
        ).await.unwrap();
        let _hello: ServerMessage =
            framed::read_frame_async(&mut r).await.unwrap().unwrap();

        // Two inferences in a row, alternating session ids.
        for (i, sid) in [("a", "s-a"), ("b", "s-b")].iter().enumerate() {
            framed::write_frame_async(&mut w, &ClientMessage::Inference(InferenceRequest {
                role: GpuRole::Deep,
                input: InferenceInput::Text(format!("call-{}", sid.0)),
                max_tokens: 32,
                session: SessionMode::Persistent { session_id: sid.1.into() },
                inference_config: InferenceConfig::default(),
                tools: Vec::new(),
                tool_results: Vec::new(),
                cache_hash: i as u64,
                stream: false,
                progress: false,
            })).await.unwrap();

            let reply: ServerMessage =
                framed::read_frame_async(&mut r).await.unwrap().unwrap();
            match reply {
                ServerMessage::InferenceComplete(resp) => {
                    assert_eq!(resp.text, format!("echo: call-{}", sid.0));
                    assert_eq!(resp.session_id.as_deref(), Some(sid.1));
                }
                other => panic!("call {i}: expected InferenceComplete, got {other:?}"),
            }
        }
    });
}
