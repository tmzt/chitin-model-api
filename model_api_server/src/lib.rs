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
pub mod subprocess_slot;

// Real backend — gated on `llama-cpp` because it pulls
// thinker_impl + llama-cpp-2 + llama_engine. Without the feature,
// only StubSlot is available (which is what the integration tests
// use).
#[cfg(feature = "llama-cpp")]
pub mod llama_slot;

pub use slot::{DiscardSink, SlotHandle, SlotRequest, SlotResponse, StreamSink};

// Internal — events the streaming handler shuttles from the slot
// sink to the wire-frame writer.
enum SlotEvent {
    Chunk(model_api_proto::StreamChunk),
    Progress(model_api_proto::ProgressEvent),
}

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
                // Streaming gate: when either `stream` or `progress`
                // is on, route through `slot.run_stream` with a sink
                // that pushes wire frames as the slot emits events.
                // Otherwise the cheap non-streaming path — slot
                // returns one final response, we wrap it.
                let wants_stream = req.stream || req.progress;

                let reply = if wants_stream {
                    // Sink forwards Chunk / Progress events into a
                    // bounded async channel that this fn drains.
                    // Bounded(64) so a slow JS client can't make the
                    // slot buffer unbounded — when the channel fills
                    // up the sink's send blocks (drops, technically,
                    // since try_send returns Err on full). Real
                    // backpressure needs the slot to await the
                    // sink's send, which the trait API doesn't
                    // support today; revisit if we ever see real
                    // congestion.
                    let (event_tx, event_rx) =
                        async_channel::bounded::<SlotEvent>(64);
                    struct Forward(async_channel::Sender<SlotEvent>);
                    impl crate::slot::StreamSink for Forward {
                        fn on_chunk(&self, c: model_api_proto::StreamChunk) {
                            let _ = self.0.try_send(SlotEvent::Chunk(c));
                        }
                        fn on_progress(&self, p: model_api_proto::ProgressEvent) {
                            let _ = self.0.try_send(SlotEvent::Progress(p));
                        }
                    }
                    let sink: Forward = Forward(event_tx.clone());

                    // Run the slot on a background task so the
                    // event-drain loop on this task can interleave
                    // with the slot's event emission. Without this
                    // split, sink's try_send would race against our
                    // event_rx.recv on the same task and we'd lose
                    // events at the channel boundary.
                    let slot2 = slot.clone();
                    let req_for_slot = SlotRequest::from_proto(req);
                    let slot_task = smol::spawn(async move {
                        slot2.run_stream(req_for_slot, &sink).await
                    });

                    // Drain events until the slot future finishes,
                    // forwarding each as a frame to the wire. We
                    // can't poll the slot future + the channel
                    // concurrently with a single .await + select, so
                    // use a try_recv loop with a brief yield to keep
                    // the executor happy. Final response is awaited
                    // last; any events still in the queue after the
                    // slot returns get drained before we send
                    // InferenceComplete.
                    let mut final_resp: Option<Result<SlotResponse, String>> = None;
                    loop {
                        // First try to drain any pending events.
                        match event_rx.try_recv() {
                            Ok(ev) => {
                                let frame = match ev {
                                    SlotEvent::Chunk(c) => ServerMessage::Chunk(c),
                                    SlotEvent::Progress(p) => ServerMessage::Progress(p),
                                };
                                framed::write_frame_async(&mut writer, &frame).await
                                    .map_err(|e| format!("conn {conn_id}: write event: {e}"))?;
                                continue;
                            }
                            Err(async_channel::TryRecvError::Empty) => {}
                            Err(async_channel::TryRecvError::Closed) => break,
                        }
                        // Otherwise poll the slot once; if it's done
                        // grab the result. If still running, yield
                        // so the slot task can make progress.
                        if slot_task.is_finished() {
                            final_resp = Some((&mut { slot_task }).await);
                            break;
                        }
                        smol::future::yield_now().await;
                    }
                    // Drop the sink's tx clone (we held one above via
                    // sink); close event_rx by dropping it after the
                    // last try_recv. If slot_task wasn't already
                    // awaited (e.g. channel closed first), await it.
                    let final_resp = match final_resp {
                        Some(r) => r,
                        None => {
                            // Slot completed via the channel-closed
                            // path; await its task to retrieve the
                            // Result. Channel close means the Forward
                            // sink got dropped, which only happens
                            // when slot_task completed and the
                            // closure was dropped — so the task is
                            // already finished.
                            // smol::Task::poll wouldn't compile here;
                            // a fresh await is correct because
                            // is_finished was true.
                            return Err(format!(
                                "conn {conn_id}: slot channel closed before final response"
                            ));
                        }
                    };

                    // Drain any tail events queued while we were
                    // waiting on the slot task — make sure no Chunks
                    // arrive after InferenceComplete in the wire
                    // order.
                    while let Ok(ev) = event_rx.try_recv() {
                        let frame = match ev {
                            SlotEvent::Chunk(c) => ServerMessage::Chunk(c),
                            SlotEvent::Progress(p) => ServerMessage::Progress(p),
                        };
                        framed::write_frame_async(&mut writer, &frame).await
                            .map_err(|e| format!("conn {conn_id}: write tail event: {e}"))?;
                    }

                    match final_resp {
                        Ok(SlotResponse(inf)) => ServerMessage::InferenceComplete(inf),
                        Err(e) => ServerMessage::InferenceError { message: e },
                    }
                } else {
                    // Non-streaming fast path.
                    let resp = slot.run(SlotRequest::from_proto(req)).await;
                    match resp {
                        Ok(SlotResponse(inf)) => ServerMessage::InferenceComplete(inf),
                        Err(e) => ServerMessage::InferenceError { message: e },
                    }
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
                // Polite client hangup. Close this connection only —
                // the server keeps running, the slot keeps its other
                // references. Reply with Goodbye so the client can
                // distinguish a clean disconnect from an abrupt EOF
                // (e.g. a kernel-side socket close on a crashed
                // peer). When this fn returns, the `slot: Arc<dyn
                // SlotHandle>` clone we hold drops — the slot's
                // refcount goes down by one, hits zero when the
                // last connection ends + serve()'s root Arc also
                // drops (today: never, slot lives for the server's
                // lifetime).
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
