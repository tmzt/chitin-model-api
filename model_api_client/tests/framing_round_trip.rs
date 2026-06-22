//! End-to-end framing test over a real `UnixStream` pair.
//!
//! Validates: write_frame_async on one end produces bytes that
//! read_frame_async on the other end decodes back to the original
//! message. Multiple frames in a row stream cleanly without
//! re-buffering between calls. EOF is reported as `None`, not an
//! error.

use model_api_client::framed::{read_frame_async, write_frame_async};
use model_api_proto::{
    ClientMessage, GpuRole, InferenceConfig, InferenceRequest, InferenceInput,
    ServerMessage, SessionMode, PROTOCOL_VERSION,
};

#[test]
fn round_trips_three_frames_over_unix_stream() {
    smol::block_on(async {
        let (a, b) = smol::Async::<std::os::unix::net::UnixStream>::pair().unwrap();
        let (mut a_read, mut a_write) = futures_lite::io::split(a);
        let (mut b_read, _b_write) = futures_lite::io::split(b);

        // Sender: drop 3 frames in order.
        let writer = smol::spawn(async move {
            write_frame_async(
                &mut a_write,
                &ClientMessage::Hello { protocol_version: PROTOCOL_VERSION },
            ).await.unwrap();
            write_frame_async(
                &mut a_write,
                &ClientMessage::Inference(InferenceRequest {
                    role: GpuRole::Deep,
                    input: InferenceInput::Text("hello".into()),
                    max_tokens: 64,
                    session: SessionMode::Persistent { session_id: "s-1".into() },
                    inference_config: InferenceConfig::default(),
                    cache_hash: 0xDEADBEEF,
                    stream: true,
                    progress: false,
                }),
            ).await.unwrap();
            write_frame_async(&mut a_write, &ClientMessage::Shutdown).await.unwrap();
            // Drop writer → EOF on the peer.
            drop(a_write);
        });

        // Reader: pull each frame, assert it round-tripped.
        let m1: ClientMessage = read_frame_async(&mut b_read).await.unwrap().unwrap();
        match m1 {
            ClientMessage::Hello { protocol_version } => {
                assert_eq!(protocol_version, PROTOCOL_VERSION);
            }
            other => panic!("frame 1 unexpected: {other:?}"),
        }

        let m2: ClientMessage = read_frame_async(&mut b_read).await.unwrap().unwrap();
        match m2 {
            ClientMessage::Inference(req) => {
                assert_eq!(req.role, GpuRole::Deep);
                assert_eq!(req.max_tokens, 64);
                assert_eq!(req.cache_hash, 0xDEADBEEF);
                assert!(req.stream);
                assert!(!req.progress);
                match req.session {
                    SessionMode::Persistent { session_id } => assert_eq!(session_id, "s-1"),
                    other => panic!("session_mode unexpected: {other:?}"),
                }
                match req.input {
                    InferenceInput::Text(t) => assert_eq!(t, "hello"),
                    other => panic!("input unexpected: {other:?}"),
                }
            }
            other => panic!("frame 2 unexpected: {other:?}"),
        }

        let m3: ClientMessage = read_frame_async(&mut b_read).await.unwrap().unwrap();
        assert!(matches!(m3, ClientMessage::Shutdown));

        // Writer dropped → next read is clean EOF.
        let m_eof: Option<ClientMessage> = read_frame_async(&mut b_read).await.unwrap();
        assert!(m_eof.is_none(), "expected None on EOF, got {m_eof:?}");

        // Background writer should have finished.
        writer.await;
    });
}

#[test]
fn server_messages_also_round_trip() {
    smol::block_on(async {
        let (a, b) = smol::Async::<std::os::unix::net::UnixStream>::pair().unwrap();
        let (mut a_read, mut a_write) = futures_lite::io::split(a);
        let (mut b_read, _b_write) = futures_lite::io::split(b);

        let server_hello = ServerMessage::Hello {
            protocol_version: PROTOCOL_VERSION,
            model_name: "gemma-4-26b".into(),
            gpu_memory_mb: Some(8192),
        };
        let writer = smol::spawn(async move {
            write_frame_async(&mut a_write, &server_hello).await.unwrap();
            drop(a_write);
        });

        let got: ServerMessage = read_frame_async(&mut b_read).await.unwrap().unwrap();
        match got {
            ServerMessage::Hello { protocol_version, model_name, gpu_memory_mb } => {
                assert_eq!(protocol_version, PROTOCOL_VERSION);
                assert_eq!(model_name, "gemma-4-26b");
                assert_eq!(gpu_memory_mb, Some(8192));
            }
            other => panic!("unexpected: {other:?}"),
        }

        // Also drain a clean EOF.
        let eof: Option<ServerMessage> = read_frame_async(&mut b_read).await.unwrap();
        assert!(eof.is_none());

        writer.await;
    });
}

#[test]
fn truncated_header_surfaces_as_unexpected_eof() {
    // Write only 2 bytes of a header, then close. Reader should
    // see "started reading a frame, got UnexpectedEof".
    smol::block_on(async {
        let (a, b) = smol::Async::<std::os::unix::net::UnixStream>::pair().unwrap();
        let (_a_read, mut a_write) = futures_lite::io::split(a);
        let (mut b_read, _b_write) = futures_lite::io::split(b);

        let writer = smol::spawn(async move {
            use futures_lite::io::AsyncWriteExt;
            a_write.write_all(&[0u8, 0]).await.unwrap();
            drop(a_write);
        });

        let r: std::io::Result<Option<ClientMessage>> = read_frame_async(&mut b_read).await;
        assert!(r.is_err(), "expected error from truncated header, got {r:?}");
        assert_eq!(r.unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);

        writer.await;
    });
}
