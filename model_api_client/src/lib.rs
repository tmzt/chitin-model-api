//! UDS client for the `model_api` server. Connects, exchanges the
//! `Hello` handshake, and exposes a typed async API for inference
//! plus shutdown.
//!
//! Designed so `thinker_impl::spawn()` can drop in here verbatim:
//! you get a `Sender<InferenceRequest>` shape that mirrors today's
//! in-process channel, but every send round-trips through UDS to the
//! external model_api binary.
//!
//! Wire format: `[u32 LE len][bincode payload]` — see
//! [`model_api_proto`] for the message structs.

use model_api_proto::{ClientMessage, ServerMessage, PROTOCOL_VERSION};

pub mod framed;
pub mod sync;

// Re-export the proto types so downstream callers only need to depend
// on this crate, not directly on `model_api_proto`.
pub use model_api_proto as proto;

// Re-export the sync client as the canonical surface — it's what
// the Node bindings (model_api_node), the openai-api shim, and
// anything else without a smol reactor uses. The async-shaped
// `Client` below is kept for code paths that already have a smol
// reactor running (e.g. thinker_impl's bridge thread).
pub use sync::SyncClient;

/// Errors the client can return at the API surface. Wire-level
/// failures (codec decode, socket read/write) fold into `Io`; the
/// server-side `InferenceError` envelope lives in `Server`.
#[derive(Debug)]
pub enum ClientError {
    /// Local I/O failure (socket closed, write half broken, codec
    /// decode failed). The accompanying string is best-effort context.
    Io(String),
    /// Protocol mismatch on the `Hello` handshake.
    ProtocolVersion { server: u32, client: u32 },
    /// Server replied with `InferenceError` for the most recent
    /// inference. The string is the server's `message` field.
    Server(String),
    /// Server hung up unexpectedly mid-request.
    Disconnected,
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(s) => write!(f, "io: {s}"),
            Self::ProtocolVersion { server, client } => {
                write!(f, "protocol mismatch: server={server}, client={client}")
            }
            Self::Server(s) => write!(f, "server error: {s}"),
            Self::Disconnected => write!(f, "server disconnected"),
        }
    }
}

impl std::error::Error for ClientError {}

// ── Connection ──────────────────────────────────────────────────────

/// A connected model_api session. Hold one of these per logical
/// inference channel — the server today serializes requests behind a
/// single global slot, so multiple `Client` instances offer no
/// throughput win but do let independent callers cancel each other
/// cleanly via per-connection state.
pub struct Client {
    // Connection state is intentionally a thin shell during scaffold —
    // the real codec + dispatch lives in a follow-up commit. Held as
    // an Option so `connect()` returns a Client value even before the
    // codec is wired.
    _socket_path: std::path::PathBuf,
}

impl Client {
    /// Connect to the model_api server at `socket_path`. Performs the
    /// `Hello` handshake and verifies the protocol version. Returns
    /// the connected client.
    ///
    /// **Scaffold stub** — wiring lives in a follow-up commit. Today
    /// this just records the socket path so call sites can compile
    /// and unit-test against the API shape.
    pub async fn connect(socket_path: impl Into<std::path::PathBuf>) -> Result<Self, ClientError> {
        let socket_path = socket_path.into();
        log::info!(
            "[model_api_client] connect stub: socket={} (protocol_version={})",
            socket_path.display(),
            PROTOCOL_VERSION,
        );
        Ok(Self { _socket_path: socket_path })
    }

    /// Submit an inference request. **Scaffold stub.** Returns an
    /// error today so callers wiring this in early are forced to
    /// notice when the real implementation lands.
    pub async fn inference(
        &self,
        _req: proto::InferenceRequest,
    ) -> Result<InferenceStream, ClientError> {
        Err(ClientError::Io("model_api_client: inference() not implemented yet".into()))
    }

    /// Best-effort cancel of the most recent in-flight inference.
    /// **Scaffold stub.**
    pub async fn cancel(&self) -> Result<(), ClientError> {
        let _ = ClientMessage::Cancel;
        Ok(())
    }

    /// Close this client's connection cleanly. Does NOT shut down
    /// the server — only decrements the slot's refcount by one;
    /// other clients continue. **Scaffold stub** (use
    /// [`SyncClient::shutdown`] for the real implementation).
    pub async fn shutdown(&self) -> Result<(), ClientError> {
        let _ = ClientMessage::Shutdown;
        let _ = ServerMessage::Goodbye;
        Ok(())
    }
}

/// Returned by [`Client::inference`]. A receiver for streamed chunks
/// + progress and a final `recv_final` that yields the
/// [`proto::InferenceResponse`] when the server says `InferenceComplete`.
///
/// **Scaffold stub** — concrete channel wiring lives in a follow-up
/// commit.
pub struct InferenceStream {
    _marker: (),
}
