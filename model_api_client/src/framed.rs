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

use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
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

/// Encode `msg` and write the whole frame to `w`. Buffers the
/// payload in memory (the `encode` step needs the full length up
/// front to write the header), then drops it in a single
/// `write_all`. Caller is responsible for the writer's flush
/// semantics — UDS doesn't auto-flush across `write_all` so call
/// `w.flush().await` between bursts when latency matters.
pub async fn write_frame_async<W, T>(w: &mut W, msg: &T) -> Result<(), io::Error>
where
    W: AsyncWriteExt + Unpin,
    T: Serialize,
{
    let frame = encode(msg)?;
    w.write_all(&frame).await
}

/// Read exactly one frame from `r`. Reads the 4-byte header, then
/// the payload, then bincode-decodes. Returns `Ok(None)` on clean
/// EOF (the peer closed before sending a header); any other read
/// short of the expected length surfaces as `UnexpectedEof`.
///
/// Allocates one `Vec<u8>` per call sized to the payload length.
/// Frames are small (handshake / progress) to medium (inference
/// response with raw_text in the kilobytes), so the allocation is
/// not a hot-path concern. If it becomes one, lift the buffer onto
/// the connection state.
pub async fn read_frame_async<R, T>(r: &mut R) -> Result<Option<T>, io::Error>
where
    R: AsyncReadExt + Unpin,
    T: DeserializeOwned,
{
    let mut hdr = [0u8; 4];
    // Distinguish clean EOF from a short header. Read into the buffer
    // one byte at a time for the first byte so we can detect the EOF
    // boundary cleanly without `read_exact`'s "fail on any short
    // read" treatment.
    let first = match r.read(&mut hdr[..1]).await? {
        0 => return Ok(None),
        1 => 1,
        _ => unreachable!("read of buf[..1] returned > 1"),
    };
    if first < 4 {
        // Got at least one byte — anything short now is an unclean
        // EOF (peer closed mid-header).
        r.read_exact(&mut hdr[first..]).await?;
    }
    let len = u32::from_le_bytes(hdr) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame header {len} > MAX_FRAME_BYTES {MAX_FRAME_BYTES}"),
        ));
    }
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload).await?;
    let value: T = bincode::deserialize(&payload)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bincode decode: {e}")))?;
    Ok(Some(value))
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
