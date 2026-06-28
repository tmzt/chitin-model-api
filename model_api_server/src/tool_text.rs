//! Text-marker utilities for slot backends that don't have native
//! tool support (llama.cpp via thinker_impl today; future backends
//! that just take/return raw strings).
//!
//! Belt-and-suspenders: a slot whose backend natively returns
//! structured tool calls (e.g. LiteRT-LM Conversation with
//! `set_tools` + JSON responses) MUST NOT call these — its native
//! path is the source of truth. These helpers exist for the
//! raw-text path where the model emits tool-call markers inline
//! with the response and we have to scan for them.

use common::tool_format::ToolFormat;
use model_api_proto::{Role, ToolCall, ToolResult, Turn};

/// Wrap one tool result via the active model's `ToolFormat`.
/// Output is the format's `format_response(...)` (e.g. Gemma 4
/// wraps as `<|tool_response>{output}<tool_response|>`). The
/// caller decides where to splice the result string in.
pub fn format_tool_result_text(result: &ToolResult, tool_format: &dyn ToolFormat) -> String {
    tool_format.format_response(&result.output)
}

/// Convert a list of tool results into `Turn`s with `Role::Tool`,
/// each one's `content` already wrapped by the model's
/// `tool_format`. Use this when a caller has supplied `tool_results`
/// alongside an `InferenceInput::Turns` request — the slot can
/// append these turns to the turn list before rendering. The
/// `tool_call_id` is propagated so future ToolFormats that need
/// correlation can use it.
pub fn tool_results_to_turns(
    results: &[ToolResult],
    tool_format: &dyn ToolFormat,
) -> Vec<Turn> {
    results
        .iter()
        .map(|r| Turn {
            role: Role::Tool,
            content: format_tool_result_text(r, tool_format),
            tool_call_id: Some(r.call_id.clone()),
        })
        .collect()
}

/// Legacy Text-path prepend: produce `format_response(r1) +
/// format_response(r2) + ... + user_text`. This is what
/// `LlamaSlot` did before the Turns refactor — kept here so the
/// `InferenceInput::Text` callers continue to work unchanged.
pub fn prepend_tool_results_to_text(
    user_text: &str,
    results: &[ToolResult],
    tool_format: &dyn ToolFormat,
) -> String {
    if results.is_empty() {
        return user_text.to_string();
    }
    let mut out = String::new();
    for r in results {
        out.push_str(&format_tool_result_text(r, tool_format));
        out.push('\n');
    }
    out.push_str(user_text);
    out
}

/// Scan `text` for tool-call marker pairs and convert each body
/// into a wire [`ToolCall`]. Uses `tool_format.open_marker()` /
/// `close_marker()` to locate pairs and `parse_body` to extract
/// `(name, args)`. Tool ids are positional (`tc-0`, `tc-1`, …) —
/// the underlying ToolFormat doesn't surface ids; clients should
/// echo whatever id they received in the matching `ToolResult`.
pub fn extract_tool_calls(text: &str, tool_format: &dyn ToolFormat) -> Vec<ToolCall> {
    let open = tool_format.open_marker();
    let close = tool_format.close_marker();
    if open.is_empty() || close.is_empty() {
        return Vec::new();
    }
    let mut calls = Vec::new();
    let mut cursor = 0usize;
    while let Some(start) = text[cursor..].find(open) {
        let body_start = cursor + start + open.len();
        let Some(end) = text[body_start..].find(close) else { break };
        let body = &text[body_start..body_start + end];
        cursor = body_start + end + close.len();
        if let Some(parsed) = tool_format.parse_body(body) {
            // `Gemma4ToolFormat::parse_body` (and other in-process
            // formats) inject a redundant `tool: <name>` key into
            // args as a sentinel for the in-process
            // EpiphanyDispatcher. Strip it at the wire boundary so
            // remote callers (TypeBox validators, OpenAI-style
            // function callers) don't reject the unknown field.
            let mut args = parsed.args;
            if let Some(obj) = args.as_object_mut() {
                obj.remove("tool");
            }
            let args_json = serde_json::to_string(&args)
                .unwrap_or_else(|_| "{}".to_string());
            calls.push(ToolCall {
                id: format!("tc-{}", calls.len()),
                name: parsed.name,
                arguments_json: args_json,
            });
        }
    }
    calls
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::tool_format::Gemma4ToolFormat;

    #[test]
    fn extract_zero_calls_from_plain_text() {
        let calls = extract_tool_calls("the answer is 4.", &Gemma4ToolFormat);
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn extract_one_call_gemma4_shape() {
        let text = "reasoning… <|tool_call>call:calculator{\"expr\":\"2+2\"}<tool_call|> done";
        let calls = extract_tool_calls(text, &Gemma4ToolFormat);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "calculator");
        assert_eq!(calls[0].id, "tc-0");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].arguments_json).unwrap();
        assert_eq!(args.get("expr").and_then(|v| v.as_str()), Some("2+2"));
    }

    #[test]
    fn extract_multiple_calls_keep_order_and_unique_ids() {
        let text = "\
            <|tool_call>call:a{\"k\":1}<tool_call|>\n\
            interlude\n\
            <|tool_call>call:b{\"k\":2}<tool_call|>";
        let calls = extract_tool_calls(text, &Gemma4ToolFormat);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].id, "tc-1");
    }

    #[test]
    fn tool_results_to_turns_wraps_each_with_format_response() {
        let results = vec![
            ToolResult { call_id: "tc-0".into(), output: "42".into() },
            ToolResult { call_id: "tc-1".into(), output: "ok".into() },
        ];
        let turns = tool_results_to_turns(&results, &Gemma4ToolFormat);
        assert_eq!(turns.len(), 2);
        assert!(matches!(turns[0].role, Role::Tool));
        assert!(turns[0].content.contains("42"));
        assert!(turns[0].content.contains("tool_response"));
        assert_eq!(turns[0].tool_call_id.as_deref(), Some("tc-0"));
    }

    #[test]
    fn prepend_empty_results_passes_through() {
        let s = prepend_tool_results_to_text("hi", &[], &Gemma4ToolFormat);
        assert_eq!(s, "hi");
    }
}
