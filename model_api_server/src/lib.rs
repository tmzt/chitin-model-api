//! Library half of `chitin-model-api`. Holds everything the binary
//! needs but in testable form: connection handler, slot abstraction,
//! request bridge. The binary is a thin shim around `serve()`.
//!
//! Slot abstraction: instead of hard-coding a llama.cpp dependency
//! into the connection handler, the server takes a `SlotHandle`
//! trait object. Production wires it up to
//! `thinker_impl::spawn_resource`; tests inject a stub that echoes
//! requests back without loading a model.

use std::sync::Arc;

use futures_lite::io::AsyncWriteExt;
use model_api_proto::{
    ClientMessage, InferenceResponse, ServerMessage, PROTOCOL_VERSION,
};

pub mod framed;
pub mod slot;

// Real backend — gated on `llama-cpp` because it pulls
// thinker_impl + llama-cpp-2 + llama_engine. Without the feature,
// only StubSlot is available (which is what the integration tests
// use).
#[cfg(feature = "llama-cpp")]
pub mod llama_slot;

pub use slot::{SlotHandle, SlotRequest, SlotResponse};

/// Per-connection identity for logging.
static CONN_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn next_conn_id() -> u64 {
    CONN_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Listen on `socket_path`, accept connections forever, hand each to
/// `handle_client`. Removes a stale socket file on start.
///
/// Returns only on `accept()` failure (or a SIGTERM-shaped cancel
/// from outside).
pub async fn serve(
    socket_path: std::path::PathBuf,
    slot: Arc<dyn SlotHandle>,
) -> Result<(), String> {
    let _ = std::fs::remove_file(&socket_path);

    let listener = smol::net::unix::UnixListener::bind(&socket_path)
        .map_err(|e| format!("bind {}: {e}", socket_path.display()))?;
    log::info!("[model_api] listening on {}", socket_path.display());

    loop {
        let (stream, _addr) = listener
            .accept()
            .await
            .map_err(|e| format!("accept: {e}"))?;
        let conn_id = next_conn_id();
        log::info!("[model_api] conn {conn_id} accepted");
        let slot = slot.clone();
        smol::spawn(async move {
            if let Err(e) = handle_client(conn_id, stream, slot).await {
                log::warn!("[model_api] conn {conn_id} ended: {e}");
            } else {
                log::info!("[model_api] conn {conn_id} closed cleanly");
            }
        })
        .detach();
    }
}

/// Per-connection loop. Handshake first, then a request/response loop
/// until the peer closes or sends `Shutdown` (which today shuts down
/// only this connection — process-wide drain lives in a future
/// commit alongside the supervisor wiring).
pub async fn handle_client(
    conn_id: u64,
    stream: smol::net::unix::UnixStream,
    slot: Arc<dyn SlotHandle>,
) -> Result<(), String> {
    let (mut reader, mut writer) = futures_lite::io::split(stream);

    // ── Handshake ───────────────────────────────────────────────────
    // First frame must be ClientMessage::Hello. Reply with our own
    // Hello and the loaded model's metadata.
    let first: Option<ClientMessage> =
        framed::read_frame_async(&mut reader).await
            .map_err(|e| format!("conn {conn_id}: read hello: {e}"))?;
    match first {
        Some(ClientMessage::Hello { protocol_version }) => {
            if protocol_version != PROTOCOL_VERSION {
                let err = format!(
                    "protocol mismatch: client={protocol_version} server={PROTOCOL_VERSION}"
                );
                let _ = framed::write_frame_async(
                    &mut writer,
                    &ServerMessage::InferenceError { message: err.clone() },
                ).await;
                return Err(err);
            }
        }
        Some(other) => {
            return Err(format!("conn {conn_id}: expected Hello, got {other:?}"));
        }
        None => return Err(format!("conn {conn_id}: EOF before Hello")),
    }
    framed::write_frame_async(
        &mut writer,
        &ServerMessage::Hello {
            protocol_version: PROTOCOL_VERSION,
            model_name: slot.model_name().to_string(),
            gpu_memory_mb: slot.gpu_memory_mb(),
        },
    ).await
        .map_err(|e| format!("conn {conn_id}: write hello: {e}"))?;

    // ── Request loop ────────────────────────────────────────────────
    loop {
        let msg: Option<ClientMessage> =
            framed::read_frame_async(&mut reader).await
                .map_err(|e| format!("conn {conn_id}: read: {e}"))?;
        let Some(msg) = msg else {
            // Clean EOF.
            return Ok(());
        };
        match msg {
            ClientMessage::Hello { .. } => {
                return Err(format!("conn {conn_id}: duplicate Hello"));
            }
            ClientMessage::Inference(req) => {
                // Hand off to the slot — single global slot today, so
                // calls serialize naturally. The slot returns either
                // a final response or an error string.
                let stream_progress = req.progress;
                let _stream_chunks = req.stream;
                let _ = stream_progress; // Wired in a follow-up commit
                let _ = _stream_chunks;  // — for now we just send the
                                         // final InferenceComplete.

                let resp = slot.run(SlotRequest::from_proto(req)).await;
                let reply = match resp {
                    Ok(SlotResponse(inf)) => ServerMessage::InferenceComplete(inf),
                    Err(e) => ServerMessage::InferenceError { message: e },
                };
                framed::write_frame_async(&mut writer, &reply).await
                    .map_err(|e| format!("conn {conn_id}: write reply: {e}"))?;
                let _ = writer.flush().await;
            }
            ClientMessage::Cancel => {
                // No-op today — cancellation requires plumbing into
                // the slot's request loop, which lands with stage 3.
                log::debug!("[model_api] conn {conn_id}: Cancel (no-op for now)");
            }
            ClientMessage::Shutdown => {
                framed::write_frame_async(&mut writer, &ServerMessage::Goodbye).await
                    .map_err(|e| format!("conn {conn_id}: write goodbye: {e}"))?;
                let _ = writer.flush().await;
                return Ok(());
            }
        }
    }
}

/// Convenience wrapper exported for the binary so it doesn't need to
/// know about `InferenceResponse` directly when constructing a
/// boilerplate error reply.
pub fn error_response(message: impl Into<String>) -> InferenceResponse {
    InferenceResponse {
        text: format!("[error] {}", message.into()),
        session_id: None,
        raw_text: None,
        injections: Vec::new(),
        tool_calls: Vec::new(),
        replacement: None,
    }
}
