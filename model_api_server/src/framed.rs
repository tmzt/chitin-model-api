//! Server-side mirror of `model_api_client::framed`. Same wire shape
//! (`[u32 LE len][bincode payload]`), same `MAX_FRAME_BYTES` cap, so
//! both sides reject the same oversized headers.
//!
//! Kept as its own file (not a shared crate) for now because the
//! codec is ~50 lines and the client/server lifecycle is divergent
//! enough that a single util crate would mostly carry duplicated
//! comments. If a third consumer appears, lift this into a shared
//! `model_api_codec` crate.

use std::io;

use serde::{de::DeserializeOwned, Serialize};

pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

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
