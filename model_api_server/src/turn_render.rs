//! Turn-list → chat-template string renderer for the LlamaSlot path.
//!
//! `model_api_proto::Turn` is the wire shape; `gemma_utils::chat_format::ChatFormat`
//! is the per-model template (Gemma4Format, ChatMLFormat, …). This
//! module is the join point — a free function that loops a Turn list
//! through the format's per-role wrappers and tacks the
//! `format_assistant_start()` priming on the end so the model
//! continues into a new assistant turn.
//!
//! Lives in `model_api_server` (not `common`) because `common` is a
//! leaf crate that intentionally doesn't depend on `model_api_proto`
//! — only the slot ever needs to join the two.

use gemma_utils::chat_format::ChatFormat;
use model_api_proto::{Role, Turn};

/// Render a turn list using the model's `ChatFormat`. Returns a
/// fully-templated prompt string ready to feed straight to the
/// underlying inference call. The closing
/// `format_assistant_start()` primes the model to emit a new
/// assistant turn — callers SHOULD NOT add their own.
///
/// `system_prompt`, when set, is prepended as a System turn unless
/// `turns` already opens with one (avoid duplicating). Tool turns
/// are rendered the same way user turns are — model-side ToolFormat
/// marker scanning happens after generation, not here.
pub fn render_turns(
    fmt: &dyn ChatFormat,
    turns: &[Turn],
    system_prompt: Option<&str>,
) -> String {
    let mut out = String::with_capacity(
        turns.iter().map(|t| t.content.len() + 32).sum::<usize>() + 64,
    );

    let already_has_system = turns
        .first()
        .map(|t| matches!(t.role, Role::System))
        .unwrap_or(false);
    if let Some(sys) = system_prompt {
        if !already_has_system && !sys.is_empty() {
            out.push_str(&fmt.format_system(sys));
        }
    }

    for turn in turns {
        match turn.role {
            Role::System => out.push_str(&fmt.format_system(&turn.content)),
            Role::User => out.push_str(&fmt.format_user(&turn.content)),
            Role::Assistant => out.push_str(&fmt.format_assistant(&turn.content)),
            // Tool turns get rendered as user-shaped messages —
            // the response wrapper (e.g. Gemma4's
            // `<|tool_response>…<tool_response|>`) is the caller's
            // job and should already be baked into `turn.content`.
            // See `tool_text::format_tool_results_into_text`.
            Role::Tool => out.push_str(&fmt.format_user(&turn.content)),
        }
    }

    out.push_str(&fmt.format_assistant_start());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemma_utils::chat_format::Gemma4Format;

    #[test]
    fn renders_simple_user_turn() {
        let s = render_turns(&Gemma4Format, &[Turn::user("hi")], None);
        // Gemma4: <|turn>user\nhi<turn|>\n<|turn>model\n
        assert_eq!(s, "<|turn>user\nhi<turn|>\n<|turn>model\n");
    }

    #[test]
    fn renders_multi_turn_with_assistant_priming() {
        let s = render_turns(
            &Gemma4Format,
            &[
                Turn::user("ask 1"),
                Turn::assistant("answer 1"),
                Turn::user("ask 2"),
            ],
            None,
        );
        assert!(s.ends_with("<|turn>model\n"));
        assert!(s.contains("<|turn>user\nask 1<turn|>"));
        assert!(s.contains("<|turn>model\nanswer 1<turn|>"));
        assert!(s.contains("<|turn>user\nask 2<turn|>"));
    }

    #[test]
    fn explicit_system_prompt_prepended_when_turns_dont_lead_with_one() {
        let s = render_turns(&Gemma4Format, &[Turn::user("hi")], Some("be brief"));
        assert!(s.starts_with("<|turn>system\nbe brief<turn|>"));
    }

    #[test]
    fn explicit_system_prompt_skipped_when_turns_already_start_with_system() {
        let s = render_turns(
            &Gemma4Format,
            &[Turn::system("baked"), Turn::user("hi")],
            Some("override"),
        );
        // System prompt arg dropped; only the in-list system turn appears.
        assert!(s.starts_with("<|turn>system\nbaked<turn|>"));
        assert!(!s.contains("override"));
    }
}
