//! End-to-end framing test over a real `UnixStream` pair.
//!
//! Validates: write_frame_async on one end produces bytes that
//! read_frame_async on the other end decodes back to the original
//! message. Multiple frames in a row stream cleanly without
//! re-buffering between calls. EOF is reported as `None`, not an
//! error.
//!
//! Note on EOF: `futures_lite::io::split` returns half-pairs that
//! share the underlying FD via an Arc. Dropping just the WriteHalf
//! does NOT close the FD — the ReadHalf binding keeps it open. To
//! get a clean EOF on the peer, drop the whole `Async<UnixStream>`
//! (don't split the writer-only side). The reader side splits cheaply
//! because we never write to it.
//!
//! All concurrency lives inside a single `smol::block_on` via
//! `futures_lite::future::zip` — `smol::spawn` would shove the
//! second half onto the global thread pool, and Rust's parallel
//! test runner can starve that pool when multiple `block_on` calls
//! grab every thread for themselves.

use futures_lite::future::zip;
use futures_lite::io::AsyncWriteExt;
use model_api_client::framed::{read_frame_async, write_frame_async};
use model_api_proto::{
    ClientMessage, GpuRole, InferenceConfig, InferenceRequest, InferenceInput,
    ServerMessage, SessionMode, PROTOCOL_VERSION,
};

#[test]
fn round_trips_three_frames_over_unix_stream() {
    smol::block_on(async {
        let (mut a, b) = smol::Async::<std::os::unix::net::UnixStream>::pair().unwrap();
        let (mut b_read, _b_write) = futures_lite::io::split(b);

        let writer = async {
            write_frame_async(
                &mut a,
                &ClientMessage::Hello { protocol_version: PROTOCOL_VERSION },
            ).await.unwrap();
            write_frame_async(
                &mut a,
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
            write_frame_async(&mut a, &ClientMessage::Shutdown).await.unwrap();
            // Drop the whole `a` — closes both halves of the FD,
            // peer sees EOF cleanly.
            drop(a);
        };

        let reader = async {
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
        };

        zip(writer, reader).await;
    });
}

#[test]
fn server_messages_also_round_trip() {
    smol::block_on(async {
        let (mut a, b) = smol::Async::<std::os::unix::net::UnixStream>::pair().unwrap();
        let (mut b_read, _b_write) = futures_lite::io::split(b);

        let server_hello = ServerMessage::Hello {
            protocol_version: PROTOCOL_VERSION,
            model_name: "gemma-4-26b".into(),
            gpu_memory_mb: Some(8192),
        };

        let writer = async {
            write_frame_async(&mut a, &server_hello).await.unwrap();
            drop(a);
        };

        let reader = async {
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
        };

        zip(writer, reader).await;
    });
}

#[test]
fn truncated_header_surfaces_as_unexpected_eof() {
    // Write only 2 bytes of a header, then close. Reader should
    // see "started reading a frame, got UnexpectedEof".
    smol::block_on(async {
        let (mut a, b) = smol::Async::<std::os::unix::net::UnixStream>::pair().unwrap();
        let (mut b_read, _b_write) = futures_lite::io::split(b);

        let writer = async {
            a.write_all(&[0u8, 0]).await.unwrap();
            drop(a);
        };

        let reader = async {
            let r: std::io::Result<Option<ClientMessage>> = read_frame_async(&mut b_read).await;
            assert!(r.is_err(), "expected error from truncated header, got {r:?}");
            assert_eq!(r.unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
        };

        zip(writer, reader).await;
    });
}
