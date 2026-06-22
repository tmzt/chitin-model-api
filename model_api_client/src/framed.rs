//! Length-prefixed bincode codec for UDS framing.
//!
//! Wire shape: `[u32 LE len][bincode payload]`. Both sides agree on
//! the codec via the proto crate; the server uses an identical impl
//! mirrored under `model_api_server::framed`.
//!
//! Length is in bytes of the bincode payload. Max length is checked
//! against [`MAX_FRAME_BYTES`] so a corrupt header can't make us
//! allocate gigabytes.

use std::io;

use serde::{de::DeserializeOwned, Serialize};

/// Hard cap on a single frame's payload. Inference responses with
/// long `raw_text` can run into the megabytes; bumped well above that
/// but well below "would OOM us". Bump if a legit producer hits it.
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Encode `msg` as a length-prefixed bincode frame.
pub fn encode<T: Serialize>(msg: &T) -> Result<Vec<u8>, io::Error> {
    let payload = bincode::serialize(msg)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bincode encode: {e}")))?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame {} > MAX_FRAME_BYTES {}", payload.len(), MAX_FRAME_BYTES),
        ));
    }
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Decode a single frame from `buf`. Returns the decoded value and
/// the number of bytes consumed; caller can drop those from the buffer.
/// `None` means "need more bytes" — not an error.
pub fn try_decode<T: DeserializeOwned>(buf: &[u8]) -> Result<Option<(T, usize)>, io::Error> {
    if buf.len() < 4 { return Ok(None); }
    let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame header {len} > MAX_FRAME_BYTES {MAX_FRAME_BYTES}"),
        ));
    }
    if buf.len() < 4 + len { return Ok(None); }
    let value: T = bincode::deserialize(&buf[4..4 + len])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bincode decode: {e}")))?;
    Ok(Some((value, 4 + len)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_api_proto::{ClientMessage, PROTOCOL_VERSION};

    #[test]
    fn round_trips_hello() {
        let msg = ClientMessage::Hello { protocol_version: PROTOCOL_VERSION };
        let frame = encode(&msg).unwrap();
        let (got, consumed): (ClientMessage, usize) = try_decode(&frame).unwrap().unwrap();
        assert_eq!(consumed, frame.len());
        match got {
            ClientMessage::Hello { protocol_version } => {
                assert_eq!(protocol_version, PROTOCOL_VERSION);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn partial_frame_returns_none() {
        let msg = ClientMessage::Hello { protocol_version: PROTOCOL_VERSION };
        let frame = encode(&msg).unwrap();
        // Header only — too short for payload.
        let head = &frame[..4];
        let r: Result<Option<(ClientMessage, usize)>, _> = try_decode(head);
        assert!(matches!(r, Ok(None)));
        // Header + 1 byte — still partial.
        let mid = &frame[..5];
        let r: Result<Option<(ClientMessage, usize)>, _> = try_decode(mid);
        assert!(matches!(r, Ok(None)));
    }

    #[test]
    fn rejects_oversized_header() {
        let mut buf = vec![0u8; 4];
        buf[0..4].copy_from_slice(&((MAX_FRAME_BYTES as u32 + 1).to_le_bytes()));
        let r: Result<Option<(ClientMessage, usize)>, _> = try_decode(&buf);
        assert!(r.is_err());
    }
}
