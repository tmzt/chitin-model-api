//! Pluggable chat template formatting trait.
//!
//! The trait lives in `common` so both `thinker_engine` and `skills_engine`
//! can use it without cross-dependencies. Concrete implementations (ChatML,
//! Llama3, Raw, etc.) live in `thinker_engine`.

/// Trait for formatting multi-turn prompts in a model-specific template.
pub trait ChatFormat: Send + Sync {
    /// Wrap a system message in the model's template.
    fn format_system(&self, content: &str) -> String;
    /// Wrap a user message in the model's template.
    fn format_user(&self, content: &str) -> String;
    /// Wrap a complete assistant message in the model's template.
    fn format_assistant(&self, content: &str) -> String;
    /// Opening tag/prefix before generation (no closing tag — the model generates into this).
    fn format_assistant_start(&self) -> String;
}

/// Format a single turn (user + assistant) using the given format.
pub fn format_turn(fmt: &dyn ChatFormat, user: &str, assistant: &str) -> String {
    let mut out = fmt.format_user(user);
    out.push_str(&fmt.format_assistant(assistant));
    out
}

// ── Built-in format implementations ─────────────────────────────────────

/// ChatML format (Qwen, Yi).
pub struct ChatMLFormat;

impl ChatFormat for ChatMLFormat {
    fn format_system(&self, c: &str) -> String { format!("<|im_start|>system\n{c}<|im_end|>\n") }
    fn format_user(&self, c: &str) -> String { format!("<|im_start|>user\n{c}<|im_end|>\n") }
    fn format_assistant(&self, c: &str) -> String { format!("<|im_start|>assistant\n{c}<|im_end|>\n") }
    fn format_assistant_start(&self) -> String { "<|im_start|>assistant\n".to_string() }
}

/// Raw plaintext format (fallback).
pub struct RawFormat;

impl ChatFormat for RawFormat {
    fn format_system(&self, c: &str) -> String { format!("{c}\n\n") }
    fn format_user(&self, c: &str) -> String { format!("User: {c}\n\n") }
    fn format_assistant(&self, c: &str) -> String { format!("Assistant: {c}\n\n") }
    fn format_assistant_start(&self) -> String { "Assistant:".to_string() }
}

/// Gemma 4 chat template. Markers derived from the 26B-A4B-it GGUF's
/// embedded jinja: `<|turn>{role}\n{content}<turn|>\n` per message,
/// `<|turn>model\n` to open the assistant turn for generation.
/// Thinking stays on (the deep thinker IS the reasoning model — we
/// don't emit the `<|channel>thought\n<channel|>` thinking-off
/// suppression marker the template uses when `enable_thinking=false`).
pub struct Gemma4Format;

impl ChatFormat for Gemma4Format {
    fn format_system(&self, c: &str) -> String { format!("<|turn>system\n{c}<turn|>\n") }
    fn format_user(&self, c: &str) -> String { format!("<|turn>user\n{c}<turn|>\n") }
    fn format_assistant(&self, c: &str) -> String { format!("<|turn>model\n{c}<turn|>\n") }
    fn format_assistant_start(&self) -> String { "<|turn>model\n".to_string() }
}
