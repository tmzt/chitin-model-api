//! Gemma-4 (and sibling) chat-template + tool-call marker utilities.
//!
//! Provides the minimum surface `model_api_server` needs to render a
//! multi-turn prompt for llama.cpp and to detect / extract tool-call
//! markers back out of the model's raw text output. Client-side
//! consumers that also want to *render* tool catalogs into the system
//! prompt (`format_external_tools_for_*`) live outside this crate —
//! they need a tools-registry abstraction that isn't a wire concern.

pub mod chat_format;
pub mod json_lenient;
pub mod tool_format;

pub use chat_format::{ChatFormat, ChatMLFormat, Gemma4Format, RawFormat, format_turn};
pub use tool_format::{
    ChatMLToolFormat, Gemma4ToolFormat, ParsedCall, ToolFormat, XmlToolFormat,
};
