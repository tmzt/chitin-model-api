//! Parse-side of the tool-call wire formats.
//!
//! Server-side utility for detecting and extracting tool-call bodies
//! from raw model output. Client-side rendering (system-prompt
//! catalog fragments, gate-response formatting) is not covered here
//! — those need a tool-registry abstraction that isn't part of the
//! wire path.
//!
//! Three impls today:
//!   - [`XmlToolFormat`]    — legacy `<tool_call>{"tool":"NAME", ...args}</tool_call>` shape
//!   - [`ChatMLToolFormat`] — Qwen / OpenAI function-calling shape `{"name":"NAME","arguments":{...}}` inside `<tool_call>`
//!   - [`Gemma4ToolFormat`] — Gemma 4 native `<|tool_call>call:NAME{ARGS_JSON}<tool_call|>` (matches Gemma4Format chat template)

use serde_json::Value;

use crate::json_lenient::lenient_parse_object;

/// Parsed tool-call body. `args` is normalised to
/// `{"tool":NAME, ...args}` regardless of the input format so
/// downstream dispatchers read tool name + args uniformly.
#[derive(Debug, Clone)]
pub struct ParsedCall {
    pub name: String,
    pub args: Value,
}

/// Wire-format details for how a model expresses tool calls.
pub trait ToolFormat: Send + Sync {
    /// Marker the channels driver scans for to detect a tool-call
    /// open in the token stream.
    fn open_marker(&self) -> &str;

    /// Marker the channels driver scans for to detect a tool-call
    /// close. The body between `open_marker` and `close_marker` is
    /// what `parse_body` consumes.
    fn close_marker(&self) -> &str;

    /// Parse the body between markers. Returns `None` if the body
    /// doesn't match this format's syntax — callers may fall back to
    /// a permissive legacy parser to preserve back-compat with peers
    /// running a different format.
    fn parse_body(&self, body: &str) -> Option<ParsedCall>;

    /// Wrap a tool's response for injection back into the model's
    /// generation stream. The model resumes generating immediately
    /// after this block.
    fn format_response(&self, response: &str) -> String;
}

// ── XmlToolFormat (legacy `<tool_call>{...}</tool_call>` JSON-in-XML) ──

/// Legacy XML-style tool calling. Body: `{"tool":"NAME", ...args}`.
/// Response wrap: `<tool_response>...</tool_response>`.
pub struct XmlToolFormat;

impl ToolFormat for XmlToolFormat {
    fn open_marker(&self) -> &str { "<tool_call>" }
    fn close_marker(&self) -> &str { "</tool_call>" }

    fn parse_body(&self, body: &str) -> Option<ParsedCall> {
        let trimmed = body.trim();
        // Bare tool name (no JSON) — common for no-args calls.
        if !trimmed.starts_with('{') && trimmed.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Some(ParsedCall {
                name: trimmed.to_string(),
                args: serde_json::json!({ "tool": trimmed }),
            });
        }
        let mut v: Value = serde_json::from_str(trimmed).ok()?;
        let name = v
            .get("tool")
            .or_else(|| v.get("name"))
            .and_then(|x| x.as_str())?
            .to_string();
        if let Some(obj) = v.as_object_mut() {
            obj.insert("tool".into(), Value::String(name.clone()));
        }
        Some(ParsedCall { name, args: v })
    }

    fn format_response(&self, response: &str) -> String {
        format!("<tool_response>{response}</tool_response>")
    }
}

// ── ChatMLToolFormat (Qwen / OpenAI `{"name":"...","arguments":{...}}`) ──

/// Qwen / OpenAI-style tool calling. Same `<tool_call>` markers as
/// XML, but the body uses the function-calling spec shape:
/// `{"name":"NAME","arguments":{...}}`. Response wrap unchanged.
pub struct ChatMLToolFormat;

impl ToolFormat for ChatMLToolFormat {
    fn open_marker(&self) -> &str { "<tool_call>" }
    fn close_marker(&self) -> &str { "</tool_call>" }

    fn parse_body(&self, body: &str) -> Option<ParsedCall> {
        let trimmed = body.trim();
        let v: Value = serde_json::from_str(trimmed).ok()?;
        let name = v
            .get("name")
            .or_else(|| v.get("tool"))
            .and_then(|x| x.as_str())?
            .to_string();
        let mut out = serde_json::Map::new();
        out.insert("tool".into(), Value::String(name.clone()));
        if let Some(args_obj) = v.get("arguments").and_then(|a| a.as_object()) {
            for (k, val) in args_obj {
                out.insert(k.clone(), val.clone());
            }
        }
        Some(ParsedCall { name, args: Value::Object(out) })
    }

    fn format_response(&self, response: &str) -> String {
        format!("<tool_response>{response}</tool_response>")
    }
}

// ── Gemma4ToolFormat (native `<|tool_call>call:NAME{ARGS}<tool_call|>`) ──

/// Gemma 4 native tool calling. Markers `<|tool_call>call:` /
/// `<tool_call|>`; body shape `NAME{ARGS_JSON}` (no surrounding
/// quotes on the name). Response wrap `<|tool_response>...<tool_response|>`.
/// Matches the embedded jinja in `gemma-4-26B-A4B-it` GGUFs.
pub struct Gemma4ToolFormat;

impl ToolFormat for Gemma4ToolFormat {
    // Open marker is JUST `<|tool_call>` — `call:` lives inside the
    // body where `parse_body` strips it. Bare `<|tool_call>call<tool_call|>`
    // (no name) is caught here too so the dispatcher can emit an
    // `unknown tool` error back to the model rather than silently
    // dropping the call.
    fn open_marker(&self) -> &str { "<|tool_call>" }
    fn close_marker(&self) -> &str { "<tool_call|>" }

    fn parse_body(&self, body: &str) -> Option<ParsedCall> {
        // Gemma 4 occasionally leaks the special-token framing for the
        // `"` token, emitting the literal byte sequence `<|"|>` where a
        // quote character should appear. Normalise before any parse
        // attempt — even strict JSON would see `<|"|>` as a syntax
        // error otherwise. Safe to do unconditionally: the literal
        // seven-byte sequence isn't something the model would ever
        // legitimately want inside a string value.
        let normalized: String;
        let body = if body.contains(r#"<|"|>"#) {
            normalized = body.replace(r#"<|"|>"#, "\"");
            normalized.as_str()
        } else {
            body
        };

        let trimmed = body.trim();
        let stripped = trimmed.strip_prefix("call:").unwrap_or(trimmed).trim();
        if stripped.is_empty() {
            return None;
        }
        let (name, args_obj) = if let Some(brace) = stripped.find('{') {
            let name = stripped[..brace].trim().to_string();
            let raw_args = &stripped[brace..];
            // Strict JSON first; on failure, fall back to the shared
            // lenient parser. Gemma 4 frequently emits bare-string
            // tool args (`{prompt: find docs}`), and we'd rather
            // recover than silently drop the call.
            let args_json: Value = match serde_json::from_str::<Value>(raw_args) {
                Ok(v) => v,
                Err(_) => lenient_parse_object(raw_args)?,
            };
            let args_map = args_json.as_object().cloned().unwrap_or_default();
            (name, args_map)
        } else if stripped.chars().all(|c| c.is_alphanumeric() || c == '_') {
            (stripped.to_string(), serde_json::Map::new())
        } else {
            return None;
        };
        if name.is_empty() {
            return None;
        }
        let mut out = args_obj;
        out.insert("tool".into(), Value::String(name.clone()));
        Some(ParsedCall { name, args: Value::Object(out) })
    }

    fn format_response(&self, response: &str) -> String {
        format!("<|tool_response>{response}<tool_response|>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_parse_body_mixed_separators() {
        // Regression: agent runtime crashed with FatalError when
        // Gemma 4 emitted `any_of=[…],from:"…",to:"…"` — mixing `=`
        // and `:` in the same body. The lenient fallback now handles
        // both separators, so parse_body must accept the call.
        let parsed = Gemma4ToolFormat.parse_body(
            r#"call:return_query{any_of=["employment","job","hiring"],from:"2026-05-09",to:"2026-05-16"}"#,
        ).expect("mixed-separator body should parse via lenient fallback");
        assert_eq!(parsed.name, "return_query");
        assert_eq!(parsed.args["any_of"][0], "employment");
        assert_eq!(parsed.args["from"], "2026-05-09");
        assert_eq!(parsed.args["to"], "2026-05-16");
    }

    #[test]
    fn gemma4_parse_body_special_token_quote_escape() {
        // Regression: tokenizer-decoder leak — Gemma 4 emits the
        // literal byte sequence `<|"|>` in place of `"` (the
        // special-token framing for the quote token slipped through).
        // parse_body must normalise these out before the JSON parse.
        let parsed = Gemma4ToolFormat.parse_body(
            r#"call:return_query{any_of:[<|"|>job<|"|>,<|"|>employment<|"|>],from:<|"|>2026-05-10<|"|>,to:<|"|>2026-05-17<|"|>}"#,
        ).expect("special-token-escaped quotes should parse");
        assert_eq!(parsed.name, "return_query");
        assert_eq!(parsed.args["any_of"][0], "job");
        assert_eq!(parsed.args["any_of"][1], "employment");
        assert_eq!(parsed.args["from"], "2026-05-10");
        assert_eq!(parsed.args["to"], "2026-05-17");
    }
}
