//! Direct llama.cpp integration for chitin-model-api.
//!
//! This crate is a trimmed port of the parent-workspace `llama_engine`
//! that stands on its own — no `common::handles` / `thinker_helper`
//! dependencies. The wire-facing types come from `model_api_proto`;
//! chat templates + tool-call marker parsing come from `gemma_utils`.
//!
//! What's here:
//! - [`Session`] — llama.cpp `LlamaContext` wrapper (load, prefill,
//!   step, save/load KV, tokenize, detokenize). One of these per
//!   loaded model.
//! - [`prefix_cache`] — content-addressed KV cache. Save the state
//!   after a system-prompt prefill and reuse it across requests
//!   that share the same prefix.
//! - [`mtp`] — multi-token-prediction / speculative decode wrapper
//!   over `llama_rs_mtp_*` (our patches into llama-cpp-sys-2).
//! - [`prompt`] — turn-list → prompt-string rendering. Uses
//!   `gemma_utils::ChatFormat`.
//! - [`jinja_chat`] — Jinja evaluation of a GGUF's embedded
//!   `chat_template` (Gemma 4's is ~16 KB and won't compile through
//!   llama.cpp's built-in template matcher).

pub mod jinja_chat;
pub mod mtp;
pub mod prefix_cache;
pub mod prompt;
pub mod session;

pub use session::{LoadConfig, Session};
