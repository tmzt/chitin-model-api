//! Synchronous client for the model_api server.
//!
//! Wraps a `std::os::unix::net::UnixStream` with the
//! length-prefixed bincode codec. Single `Mutex` serializes all
//! requests behind one connection — that matches the single-slot
//! server, and dodges an async runtime dependency for callers
//! (Node bindings, openai-api shim, anything that already runs on
//! a thread pool).
//!
//! Each call writes a `ClientMessage` and reads `ServerMessage`s
//! until the terminal frame for that request arrives. Today we only
//! support the non-streaming path — chunks + progress events
//! arrive as `ServerMessage::Chunk` / `Progress` frames and we
//! silently drain them while waiting for `InferenceComplete`. When
//! callers need streaming, add a `inference_stream` method that
//! returns an iterator/callback over the intermediate frames.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::Mutex;

use model_api_proto::{
    ClientMessage, InferenceRequest, InferenceResponse, ProgressEvent,
    ServerMessage, StreamChunk, PROTOCOL_VERSION,
};

use crate::framed::{encode, MAX_FRAME_BYTES};
use crate::ClientError;

/// Information about the server reported in the Hello handshake.
#[derive(Debug, Clone)]
pub struct ServerInfo {
    pub protocol_version: u32,
    pub model_name: String,
    pub gpu_memory_mb: Option<u32>,
}

/// Synchronous, thread-safe model_api client.
///
/// One UDS connection, one `Mutex` around it. `inference()` /
/// `shutdown()` block the calling thread until the server replies.
/// Cloning the same physical connection isn't supported; callers that
/// want concurrent inflight requests open multiple `SyncClient`s
/// (each gets its own slot queue position).
pub struct SyncClient {
    /// Read + write share the same `UnixStream` here because the
    /// server serializes responses 1:1 with requests under our
    /// single-slot model. If a future server fan-out emits
    /// unsolicited frames, split into read/write halves + a reader
    /// thread.
    stream: Mutex<UnixStream>,
    server_info: ServerInfo,
}

impl SyncClient {
    /// Connect to the model_api server at `socket_path` and perform
    /// the Hello handshake. Returns the connected client + cached
    /// server info.
    pub fn connect<P: AsRef<Path>>(socket_path: P) -> Result<Self, ClientError> {
        let mut stream = UnixStream::connect(socket_path.as_ref())
            .map_err(|e| ClientError::Io(format!(
                "connect {}: {e}", socket_path.as_ref().display(),
            )))?;

        // Send Hello.
        write_frame(
            &mut stream,
            &ClientMessage::Hello { protocol_version: PROTOCOL_VERSION },
        )?;

        // Expect Hello back.
        let server_info = match read_frame(&mut stream)? {
            Some(ServerMessage::Hello { protocol_version, model_name, gpu_memory_mb }) => {
                if protocol_version != PROTOCOL_VERSION {
                    return Err(ClientError::ProtocolVersion {
                        server: protocol_version,
                        client: PROTOCOL_VERSION,
                    });
                }
                ServerInfo { protocol_version, model_name, gpu_memory_mb }
            }
            Some(ServerMessage::InferenceError { message }) => {
                return Err(ClientError::Server(message));
            }
            Some(other) => {
                return Err(ClientError::Io(format!(
                    "expected Hello, got {other:?}"
                )));
            }
            None => return Err(ClientError::Disconnected),
        };

        Ok(Self {
            stream: Mutex::new(stream),
            server_info,
        })
    }

    /// Cached server info from the Hello handshake.
    pub fn server_info(&self) -> &ServerInfo { &self.server_info }

    /// Submit one inference request. Blocks until the server's final
    /// `InferenceComplete` arrives. Streaming chunks + progress
    /// events are silently consumed today — see module docs for the
    /// `inference_stream` future addition.
    pub fn inference(&self, req: InferenceRequest) -> Result<InferenceResponse, ClientError> {
        let mut guard = self.stream.lock()
            .map_err(|_| ClientError::Io("client mutex poisoned".into()))?;

        write_frame(&mut *guard, &ClientMessage::Inference(req))?;

        loop {
            match read_frame(&mut *guard)? {
                Some(ServerMessage::InferenceComplete(resp)) => return Ok(resp),
                Some(ServerMessage::InferenceError { message }) => {
                    return Err(ClientError::Server(message));
                }
                Some(ServerMessage::Progress(_)) | Some(ServerMessage::Chunk(_)) => {
                    // Drained for non-streaming path; see module doc.
                }
                Some(ServerMessage::Hello { .. }) | Some(ServerMessage::Goodbye) => {
                    // Out-of-band; ignore and keep reading.
                }
                None => return Err(ClientError::Disconnected),
            }
        }
    }

    /// Submit an inference with callbacks for streaming chunks +
    /// progress events. Same semantics as [`inference`] for the
    /// final response; the callbacks fire from this thread (the one
    /// calling `inference_stream`) before each Chunk / Progress
    /// frame is dropped. Use this when the caller wants to display
    /// tokens as they arrive (e.g. a chat UI).
    pub fn inference_stream(
        &self,
        req: InferenceRequest,
        mut on_chunk: impl FnMut(StreamChunk),
        mut on_progress: impl FnMut(ProgressEvent),
    ) -> Result<InferenceResponse, ClientError> {
        let mut guard = self.stream.lock()
            .map_err(|_| ClientError::Io("client mutex poisoned".into()))?;

        write_frame(&mut *guard, &ClientMessage::Inference(req))?;

        loop {
            match read_frame(&mut *guard)? {
                Some(ServerMessage::InferenceComplete(resp)) => return Ok(resp),
                Some(ServerMessage::InferenceError { message }) => {
                    return Err(ClientError::Server(message));
                }
                Some(ServerMessage::Chunk(c)) => on_chunk(c),
                Some(ServerMessage::Progress(p)) => on_progress(p),
                Some(ServerMessage::Hello { .. }) | Some(ServerMessage::Goodbye) => {}
                None => return Err(ClientError::Disconnected),
            }
        }
    }

    /// Close this client's connection.
    ///
    /// Sends a polite `Shutdown` frame, waits for the server's
    /// `Goodbye` (or EOF), then returns. The server keeps running —
    /// this only releases this connection's slot reference; the
    /// underlying slot stays alive as long as another client holds a
    /// reference, or as long as the server process is up.
    ///
    /// Dropping the `SyncClient` without calling `shutdown` also
    /// works — the socket closes from the kernel side and the
    /// server's connection handler sees EOF. `shutdown` is the
    /// polite version that lets the server log a clean disconnect
    /// instead of an abrupt one.
    pub fn shutdown(&self) -> Result<(), ClientError> {
        let mut guard = self.stream.lock()
            .map_err(|_| ClientError::Io("client mutex poisoned".into()))?;

        write_frame(&mut *guard, &ClientMessage::Shutdown)?;

        // Drain until Goodbye or EOF.
        loop {
            match read_frame::<ServerMessage>(&mut *guard) {
                Ok(Some(ServerMessage::Goodbye)) => return Ok(()),
                Ok(Some(_)) => continue,
                Ok(None) => return Ok(()),    // server closed first; fine
                Err(ClientError::Disconnected) => return Ok(()),
                Err(e) => return Err(e),
            }
        }
    }
}

// ── Sync framing helpers ─────────────────────────────────────────────

/// Write one frame synchronously. Uses [`encode`] (the same codec
/// the async path uses) so on-wire bytes are byte-for-byte identical.
fn write_frame<W: Write, T: serde::Serialize>(w: &mut W, msg: &T) -> Result<(), ClientError> {
    let frame = encode(msg).map_err(|e| ClientError::Io(format!("encode: {e}")))?;
    w.write_all(&frame).map_err(|e| ClientError::Io(format!("write: {e}")))?;
    w.flush().map_err(|e| ClientError::Io(format!("flush: {e}")))?;
    Ok(())
}

/// Read one frame synchronously. Mirrors `read_frame_async`'s
/// EOF/short-header behaviour: returns `Ok(None)` on clean EOF
/// (server closed before header), errors on truncated header or
/// oversized payload.
fn read_frame<T: serde::de::DeserializeOwned>(
    r: &mut impl Read,
) -> Result<Option<T>, ClientError> {
    let mut hdr = [0u8; 4];
    let first = match r.read(&mut hdr[..1]) {
        Ok(0) => return Ok(None),
        Ok(1) => 1,
        Ok(n) => return Err(ClientError::Io(format!("partial header read({n})"))),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(ClientError::Io(format!("read header byte 0: {e}"))),
    };
    if first < 4 {
        r.read_exact(&mut hdr[first..])
            .map_err(|e| ClientError::Io(format!("read header tail: {e}")))?;
    }
    let len = u32::from_le_bytes(hdr) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(ClientError::Io(format!(
            "frame header {len} > MAX_FRAME_BYTES {MAX_FRAME_BYTES}"
        )));
    }
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)
        .map_err(|e| ClientError::Io(format!("read payload: {e}")))?;
    let value: T = bincode::deserialize(&payload)
        .map_err(|e| ClientError::Io(format!("bincode decode: {e}")))?;
    Ok(Some(value))
}
