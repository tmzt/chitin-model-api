//! Best-effort recovery for malformed JSON-like payloads that LLMs
//! sometimes emit and which strict `serde_json` rejects.
//!
//! Today's primary consumer is `tool_format::Gemma4ToolFormat` — Gemma
//! 4 frequently drops the quotes on bare-string tool-call arguments
//! (`{prompt: find docs}` instead of `{"prompt":"find docs"}`). Other
//! callers can reach for the same helper rather than re-rolling their
//! own tolerant parser: anywhere we eat model output that's *supposed*
//! to be JSON but might not be, prefer
//! `serde_json::from_str(...).ok().or_else(|| lenient_parse_object(...))`.
//!
//! Two recovery layers, applied in order:
//! 1. **Unquoted identifier keys** (`{prompt: "x"}`) — quote every bare
//!    identifier appearing as an object key and retry strict parse.
//!    Also normalises `key=value` separators to `key:value` (Gemma 4
//!    occasionally mixes `=` and `:` in the same body).
//!    Universal, safe, fixes most "model forgot the quotes" cases.
//! 2. **Single bare-string value** (`{prompt: find docs}`) — when (1)
//!    still doesn't parse AND the body has exactly one quoted-key
//!    entry, take everything between the colon and the closing brace
//!    as the raw string value.
//!
//! Multi-key bodies with bare string values are ambiguous (where does
//! one value end?) — the parser refuses rather than guess wrong, so
//! the caller can surface a real parse error to the model instead of
//! silently committing to a hallucinated arg split.

use serde_json::Value;

/// Best-effort recovery for a JSON object literal that strict
/// `serde_json::from_str` rejected. See module docs for the recovery
/// rules. Returns `None` when the body is ambiguous enough that we'd
/// rather fail than guess.
pub fn lenient_parse_object(raw: &str) -> Option<Value> {
    // Step 1: quote bare identifier keys.
    let quoted_keys = quote_bare_keys(raw);
    if let Ok(v) = serde_json::from_str::<Value>(&quoted_keys) {
        return Some(v);
    }

    // Step 1.5: escape literal control characters that appear inside
    // string values. Models routinely emit
    // `{"content":"line 1<actual LF>line 2"}` where the `\n` should
    // have been a JSON escape; strict serde_json rejects that with a
    // "control character in string" error. Walk the bytes, track
    // in-string state, replace LF/CR/TAB inside strings with their
    // JSON escape forms, and retry.
    let escaped = escape_string_controls(&quoted_keys);
    if escaped != quoted_keys {
        if let Ok(v) = serde_json::from_str::<Value>(&escaped) {
            return Some(v);
        }
    }

    // Step 1.7: structural over-escape rescue.
    //
    // Gemma 4 routinely emits the CLOSING structural quote of a
    // string value (and sometimes the surrounding key markers) as
    // `\"` instead of `"`, producing bodies like
    //   {"content":"…code with \"escaped\" data…\","path":"x.py"}
    // where the trailing `\"` should have closed `content`. The
    // strict parser then runs `,"path":"x.py"` into content's value.
    //
    // Bulk-replace every `\"` is wrong: it corrupts the legitimately
    // escaped data quotes inside content. The signal that
    // distinguishes structural quotes from data quotes is what's
    // ADJACENT to them in the body:
    //   * Structural OPEN  is preceded by `{`, `,`, or `:` (modulo whitespace)
    //   * Structural CLOSE is followed by `,`, `:`, or `}`  (modulo whitespace)
    //   * Data quotes are surrounded by letters / punctuation that
    //     looks like normal content, not JSON delimiters.
    // The short-key heuristic catches the most reliable case:
    // `,\"<short identifier>\":` — a new key pair starting. Both
    // surrounding `\"` are structural and the middle is too short to
    // be confused with content.
    if let Some(rewritten) = un_escape_structural_quotes(&escaped) {
        if rewritten != escaped {
            if let Ok(v) = serde_json::from_str::<Value>(&rewritten) {
                return Some(v);
            }
        }
    }

    // Step 1.7: closing-quote over-escape rescue. Gemma 4 sometimes
    // emits the CLOSING structural quotes of a string value as `\"`
    // even though the opening was bare `"`, producing bodies like:
    //   {"content":"…code with \"escaped\" quotes…\","path":"x.py"}
    // where the `\"` before `,"path":` should have been an unescaped
    // `"` closing `content`. The strict parser then runs the entire
    // suffix into content's value (since `\"` is just an escaped
    // quote, not a close). Detect the suspicious tail pattern
    // `\","KEY\":"VALUE\"}` and un-escape the structural quotes.

    // Step 2: single-key, unquoted-string-value rescue. Match
    // `{ "key": <bare string> }` where the value runs from the colon
    // to the closing brace as one unquoted string.
    let trimmed = quoted_keys.trim();
    let inner = trimmed.strip_prefix('{')?.strip_suffix('}')?.trim();
    let colon = inner.find(':')?;
    let key_part = inner[..colon].trim();
    // Reject if the key part isn't a single quoted string — multi-key
    // bodies are too ambiguous to rescue this way.
    if !key_part.starts_with('"') || !key_part.ends_with('"') { return None; }
    let key = &key_part[1..key_part.len()-1];
    if key.contains('"') { return None; }
    let value_raw = inner[colon+1..].trim();
    if value_raw.is_empty() { return None; }
    // If the value parses as JSON, prefer that — the earlier strict
    // attempt must have failed for a different reason and we shouldn't
    // double-quote a valid value.
    if let Ok(v) = serde_json::from_str::<Value>(value_raw) {
        let mut map = serde_json::Map::new();
        map.insert(key.to_string(), v);
        return Some(Value::Object(map));
    }
    // Bare value — treat as a string up to the closing brace, but
    // refuse when the value contains a top-level comma (would mean
    // it's actually a multi-key body whose later keys we'd silently
    // swallow into the first value).
    if has_top_level_comma(value_raw) { return None; }
    // Also refuse when the value STARTS with `"` — that's a (malformed)
    // JSON string literal that the strict parse already rejected.
    // Bare-string rescue is for unquoted values like
    // `{prompt: find docs}`. If the model wrote `"…` it meant a JSON
    // string; if that string didn't parse, the body is more
    // structurally damaged than this rescue can fix (e.g. Gemma 4's
    // over-escaped closing quote pattern that
    // `un_escape_overzealous_closers` targets). Returning None here
    // lets the caller surface the parse failure to the model rather
    // than silently collapsing a multi-key body into one giant
    // string.
    if value_raw.starts_with('"') { return None; }
    let mut map = serde_json::Map::new();
    map.insert(key.to_string(), Value::String(value_raw.to_string()));
    Some(Value::Object(map))
}

/// True iff `s` contains a `,` that isn't inside a string literal or
/// nested `[]` / `{}` block. Used to detect "this rescue would silently
/// absorb a sibling key" and refuse the rescue.
fn has_top_level_comma(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if b == b'\\' && i + 1 < bytes.len() { i += 2; continue; }
            if b == b'"' { in_string = false; }
            i += 1;
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' | b'{' => depth += 1,
            b']' | b'}' => depth -= 1,
            b',' if depth == 0 => return true,
            _ => {}
        }
        i += 1;
    }
    false
}

/// Walk `raw` and wrap any bare identifier appearing as an object key
/// in double quotes. Skips text inside existing string literals (and
/// handles `\"` escapes inside them) so we don't double-quote
/// anything that's already valid JSON.
///
/// Exposed publicly because it's a useful primitive on its own — e.g.
/// for log-parsing or repair tools that want the "quote bare keys"
/// half of the recovery without the bare-string-value heuristic.
pub fn quote_bare_keys(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = String::with_capacity(raw.len() + 8);
    let mut i = 0;
    let mut in_string = false;
    let mut prev_was_open_or_comma = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            out.push(b as char);
            if b == b'\\' && i + 1 < bytes.len() {
                out.push(bytes[i+1] as char);
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
            out.push('"');
            i += 1;
            prev_was_open_or_comma = false;
            continue;
        }
        if b == b'{' || b == b',' {
            out.push(b as char);
            i += 1;
            prev_was_open_or_comma = true;
            continue;
        }
        if b.is_ascii_whitespace() {
            out.push(b as char);
            i += 1;
            continue;
        }
        if prev_was_open_or_comma
            && (b.is_ascii_alphabetic() || b == b'_')
        {
            let start = i;
            while i < bytes.len() {
                let c = bytes[i];
                if c.is_ascii_alphanumeric() || c == b'_' { i += 1; } else { break; }
            }
            let ident = &raw[start..i];
            // Only quote when followed by `:` or `=` — otherwise this is
            // a bare value sitting where a key would be (or a hallucinated
            // identifier in a value position), not an object key.
            // Gemma 4 occasionally mixes separators within the same body
            // (`{any_of=[…],from:"…"}`); when the separator is `=`,
            // quote the key AND rewrite the `=` to `:` so the result
            // parses as strict JSON.
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() { j += 1; }
            let sep = bytes.get(j).copied();
            if sep == Some(b':') || sep == Some(b'=') {
                out.push('"');
                out.push_str(ident);
                out.push('"');
                if sep == Some(b'=') {
                    out.push_str(&raw[i..j]);
                    out.push(':');
                    i = j + 1;
                    prev_was_open_or_comma = false;
                    continue;
                }
            } else {
                out.push_str(ident);
            }
            prev_was_open_or_comma = false;
            continue;
        }
        out.push(b as char);
        i += 1;
        prev_was_open_or_comma = false;
    }
    out
}

/// Match the structural KV-pair pattern Gemma 4 (and similar) emits
/// at the start of every key/value after the first. The pattern is
/// **`<delim>(\\)?"<short ident>(\\)?":(\\)?"`** — a delimiter
/// (`{`, `,`, `:`, ...) followed by an OPENING quote, a short
/// identifier (~30 chars, identifier-shaped), a CLOSING quote, the
/// `:` separator, and the OPENING quote of the value. Each quote
/// position may be EITHER bare `"` OR escaped `\"` (the model
/// drifts between forms even within one body).
///
/// Whatever shape they came in, all THREE quotes in a match are
/// guaranteed structural — the middle field is too short and too
/// identifier-shaped to be data content.
///
/// Source `(?x)` allows the comment-form regex.
static STRUCTURAL_KV_PAIR: once_cell::sync::Lazy<regex::Regex> = once_cell::sync::Lazy::new(|| {
    regex::Regex::new(r#"(?x)
        (?P<delim>[\{,])    # opening delimiter
        \s*
        \\?"                 # optional-escape OPEN of key
        (?P<key>[A-Za-z_][A-Za-z0-9_.\-]{0,30})
        \\?"                 # optional-escape CLOSE of key
        \s*:\s*
        \\?"                 # optional-escape OPEN of value
    "#).expect("STRUCTURAL_KV_PAIR regex compiles")
});

/// Match the CLOSE of one value followed by the OPEN of the next
/// key. This is the `\","key\"` boundary that bridges two KV pairs.
/// Anchored: requires the prior `\\?"` to be followed by `,` and
/// then ANOTHER `\\?"<ident>\\?":` pattern — so a plain data quote
/// like `print(\"hi\", file=…)` won't match (no `key\":` follows
/// the comma).
static STRUCTURAL_VALUE_CLOSE_THEN_KEY: once_cell::sync::Lazy<regex::Regex> =
    once_cell::sync::Lazy::new(|| {
        regex::Regex::new(r#"(?x)
            \\?"                 # value's CLOSE quote
            \s*,\s*
            \\?"                 # next key's OPEN quote
            (?P<key>[A-Za-z_][A-Za-z0-9_.\-]{0,30})
            \\?"                 # next key's CLOSE quote
            \s*:
        "#).expect("STRUCTURAL_VALUE_CLOSE_THEN_KEY regex compiles")
    });

/// Match the structural CLOSER of the LAST value in the object:
/// `\\?"` (optional-escape quote) immediately followed by `}` (modulo
/// whitespace). The `\","KEY\":` case is already covered by
/// [`STRUCTURAL_KV_PAIR`]; this catches the trailing close which has
/// no following key to anchor it. We intentionally do NOT match
/// `\\?"` before `,` here — data quotes inside content are
/// frequently followed by `,` (e.g. `print(\"hi\", file=stderr)`),
/// and bulk-matching those would corrupt the content. The KV-pair
/// regex provides the anchored context that disambiguates.
static STRUCTURAL_VALUE_CLOSE: once_cell::sync::Lazy<regex::Regex> =
    once_cell::sync::Lazy::new(|| {
        regex::Regex::new(r#"\\"(\s*\})"#).expect("STRUCTURAL_VALUE_CLOSE regex compiles")
    });

/// Adjacency-based rescue for bodies where the model emitted some
/// structural quotes as `\"` instead of `"`. Match the canonical
/// `<delim>\\?"<ident>\\?":\\?"` KV-pair pattern (per
/// [`STRUCTURAL_KV_PAIR`]) plus the tail closer pattern (per
/// [`STRUCTURAL_VALUE_CLOSE`]); every quote occurrence in those
/// matches is structural and gets un-escaped. Bare quotes are
/// left alone (they're already correct). Data quotes inside
/// content — `print(\"hi\")` and friends — are surrounded by
/// letters / punctuation, not JSON delimiters, so they don't
/// match either pattern and stay escaped.
///
/// Returns the rewritten body. `None` means we found nothing to
/// rewrite. Idempotent: applying twice yields the same output.
pub fn un_escape_structural_quotes(raw: &str) -> Option<String> {
    if !raw.contains(r#"\""#) {
        // Nothing escaped to consider — early out.
        return None;
    }

    // We collect every byte range that needs rewriting (each `\"`
    // inside a structural match). Sort + dedupe + apply in one
    // pass to avoid quadratic string rewrites.
    let mut to_unescape: Vec<usize> = Vec::new();
    let collect = |m: regex::Match<'_>, out: &mut Vec<usize>| {
        // Walk the bytes of the match; each `\"` start position
        // gets added to the unescape list.
        let bytes = m.as_str().as_bytes();
        let base = m.start();
        let mut i = 0;
        while i + 1 < bytes.len() {
            if bytes[i] == b'\\' && bytes[i + 1] == b'"' {
                out.push(base + i);
                i += 2;
            } else {
                i += 1;
            }
        }
    };

    for m in STRUCTURAL_KV_PAIR.find_iter(raw) {
        collect(m, &mut to_unescape);
    }
    for m in STRUCTURAL_VALUE_CLOSE_THEN_KEY.find_iter(raw) {
        collect(m, &mut to_unescape);
    }
    for m in STRUCTURAL_VALUE_CLOSE.find_iter(raw) {
        collect(m, &mut to_unescape);
    }

    if to_unescape.is_empty() {
        return None;
    }
    to_unescape.sort_unstable();
    to_unescape.dedup();

    let mut out = String::with_capacity(raw.len());
    let mut cursor = 0usize;
    for pos in to_unescape {
        if pos < cursor {
            continue; // overlap (defensive); skip
        }
        out.push_str(&raw[cursor..pos]);
        out.push('"');
        cursor = pos + 2;
    }
    out.push_str(&raw[cursor..]);

    if out == raw {
        return None;
    }
    Some(out)
}

/// Walk `raw` and escape literal control characters (LF, CR, TAB)
/// that appear inside JSON string values. Strict serde_json rejects
/// these with "control character (\\u0000-\\u001F) found while
/// parsing a string", but LLMs (Gemma 4 especially) routinely emit
/// real newlines inside tool-call arg strings like
/// `{"content":"line 1<LF>line 2"}` where they should have been
/// JSON-escaped. Tracks in-string state with the same scanner as
/// `quote_bare_keys` so escapes outside strings (e.g. whitespace
/// between keys) are left untouched.
///
/// Returns the rewritten string. Callers should retry strict
/// `serde_json::from_str` after applying this — if no controls were
/// inside strings, the output equals the input.
pub fn escape_string_controls(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = String::with_capacity(raw.len());
    let mut i = 0;
    let mut in_string = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if b == b'\\' && i + 1 < bytes.len() {
                let nxt = bytes[i + 1];
                // Valid JSON escapes pass through unchanged.
                let is_valid_escape = matches!(
                    nxt,
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u'
                );
                if is_valid_escape {
                    out.push(b as char);
                    out.push(nxt as char);
                    i += 2;
                    continue;
                }
                // Invalid escape — model emitted a lone backslash
                // (e.g. `\w` from a regex literal in a python
                // comment). JSON requires the backslash itself be
                // escaped (`\\`), so double it up. The following
                // byte becomes a regular content character.
                out.push_str("\\\\");
                i += 1;
                continue;
            }
            match b {
                b'"' => {
                    in_string = false;
                    out.push('"');
                }
                b'\n' => out.push_str("\\n"),
                b'\r' => out.push_str("\\r"),
                b'\t' => out.push_str("\\t"),
                0x00..=0x1F => {
                    // Other C0 controls — emit as \u00XX. Rare in
                    // practice but cheap to handle.
                    out.push_str(&format!("\\u{:04x}", b));
                }
                _ => out.push(b as char),
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
            out.push('"');
            i += 1;
            continue;
        }
        out.push(b as char);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_json_passthrough() {
        let v = lenient_parse_object(r#"{"prompt":"find docs"}"#).unwrap();
        assert_eq!(v["prompt"], "find docs");
    }

    #[test]
    fn unquoted_key_quoted_value() {
        let v = lenient_parse_object(r#"{prompt: "find docs"}"#).unwrap();
        assert_eq!(v["prompt"], "find docs");
    }

    #[test]
    fn literal_newline_in_string_value() {
        // Gemma 4 emits multi-line tool-call args with REAL newline
        // bytes inside quoted string values — strict JSON rejects
        // them with "control character in string". The escape-rescue
        // path should re-encode them as `\n` and parse cleanly.
        let body = "{\"content\":\"line 1\nline 2\nline 3\",\"path\":\"a.txt\"}";
        let v = lenient_parse_object(body).expect("control-char rescue should parse");
        assert_eq!(v["content"], "line 1\nline 2\nline 3");
        assert_eq!(v["path"], "a.txt");
    }

    #[test]
    fn literal_tab_and_carriage_return_in_string() {
        let body = "{\"k\":\"col\ta\rcol b\"}";
        let v = lenient_parse_object(body).expect("control-char rescue should parse");
        assert_eq!(v["k"], "col\ta\rcol b");
    }

    #[test]
    fn escape_string_controls_leaves_clean_input_unchanged() {
        let clean = r#"{"a":1,"b":"hello"}"#;
        assert_eq!(escape_string_controls(clean), clean);
    }

    #[test]
    fn realistic_gemma4_write_tool_body() {
        // Exact shape Gemma 4 emits for a `write` tool call: multi-line
        // python source inside `content`, escaped quotes for inner
        // string literals, terminating `"path"` after the content.
        let body = "{\"content\":\"import sys\nimport re\n\ndef main():\n    print(\\\"hi\\\")\n\",\"path\":\"wordcount.py\"}";
        let v = lenient_parse_object(body).expect("realistic write body must parse");
        assert_eq!(v["path"], "wordcount.py");
        assert!(
            v["content"].as_str().unwrap().starts_with("import sys\nimport re\n"),
            "content was {:?}",
            v["content"]
        );
        assert!(v["content"].as_str().unwrap().contains("print(\"hi\")"));
    }

    #[test]
    fn gemma4_overzealous_close_quote_two_keys() {
        // Gemma 4 routinely emits the closing structural quote as
        // `\"` along with the opening of the next key, e.g.
        //   {"content":"<source>\","path":"x.py"}
        // The data quotes inside <source> are also `\"` but they're
        // surrounded by code characters, not JSON delimiters. The
        // adjacency-based rescue must un-escape ONLY the structural
        // ones (the `\"` before `,` and the `\","` pair, plus the
        // `\":` and `\"}` markers).
        let body =
            "{\"content\":\"print(\\\"hi\\\")\\nprint(\\\"bye\\\")\\\",\\\"path\\\":\\\"x.py\\\"}";
        let v = lenient_parse_object(body).expect("structural rescue must parse");
        assert_eq!(v["path"], "x.py");
        assert!(
            v["content"].as_str().unwrap().contains("print(\"hi\")"),
            "content was {:?}",
            v["content"]
        );
        assert!(v["content"].as_str().unwrap().contains("print(\"bye\")"));
    }

    #[test]
    fn structural_rescue_preserves_data_quotes() {
        // Data quotes adjacent to letters / operators must NOT be
        // un-escaped — only structural ones.
        let body = "{\"msg\":\"He said \\\"hello\\\" and left.\"}";
        let v = lenient_parse_object(body).expect("must parse");
        assert_eq!(v["msg"], "He said \"hello\" and left.");
    }

    #[test]
    fn invalid_escape_in_string_value_doubles_backslash() {
        // Model writes a python regex literal in a comment:
        //   # \w+ matches one or more chars
        // and JSON-encodes the source verbatim — the lone `\w` is
        // not a valid JSON escape. Strict serde_json rejects with
        // "invalid escape"; the control-char + invalid-escape step
        // converts `\w` → `\\w` so the literal backslash survives.
        let body = "{\"comment\":\"see \\w+ for words\"}";
        let v = lenient_parse_object(body).expect("must parse");
        assert_eq!(v["comment"], "see \\w+ for words");
    }

    /// Captured real-world failure. The body has THREE different
    /// model bugs at once: (1) literal newlines inside content,
    /// (2) over-escaped structural close quotes, (3) UNESCAPED
    /// quotes inside content (`# The prompt says "alphanumeric…"`).
    /// (1) and (2) we recover; (3) is structurally
    /// indistinguishable from a value's closing quote without
    /// reconstructing the object from structural anchors entirely.
    /// Documented as a known limitation rather than fixed.
    #[test]
    #[ignore = "model emits unescaped data quotes inside content; needs structural reconstruction, not heuristic JSON recovery"]
    fn gemma4_real_failing_body_from_disk() {
        // Captured from a live pi coding-test run. This body has:
        //   * `<|channel>thought\n<channel|>` prefix
        //   * `<|tool_call>call:write{...}<tool_call|>` frame
        //   * Inside: `{"content":"<multi-line py>","path":"x.py"}`
        //     with LITERAL newlines in content AND structural close
        //     quotes emitted as `\"`.
        let raw = include_str!("json_lenient_realbody.txt");
        let open = "<|tool_call>";
        let close = "<tool_call|>";
        let start = raw.find(open).unwrap() + open.len();
        let end = raw[start..].find(close).unwrap() + start;
        let body = raw[start..end].trim();
        // parse_body strips `call:NAME` prefix; here we just get to
        // the `{` brace.
        let brace = body.find('{').unwrap();
        let args = &body[brace..];
        let v = lenient_parse_object(args).expect("real Gemma body must parse");
        assert_eq!(
            v.get("path").and_then(|x| x.as_str()),
            Some("wordcount.py"),
            "path missing; got {:?}",
            v,
        );
        let content = v["content"].as_str().expect("content must be string");
        assert!(content.contains("import sys"), "content head: {}", &content[..50.min(content.len())]);
        assert!(content.contains("if __name__"), "content tail: {:?}", &content[content.len().saturating_sub(80)..]);
    }

    #[test]
    fn structural_rescue_mixed_bare_and_escaped() {
        // Common Gemma 4 drift: first key/value uses bare structural
        // quotes (correct), later keys come out with escaped ones.
        // The regex matches both forms at structural positions; the
        // already-bare ones stay bare, the escaped ones get
        // promoted.
        let body =
            "{\"a\":\"first\\\",\\\"b\\\":\\\"second\\\",\\\"c\\\":\\\"third\\\"}";
        let v = lenient_parse_object(body).expect("mixed bare/escaped must parse");
        assert_eq!(v["a"], "first");
        assert_eq!(v["b"], "second");
        assert_eq!(v["c"], "third");
    }

    #[test]
    fn structural_rescue_handles_three_keys() {
        let body = concat!(
            "{\"a\":\"line1\\nline2\\\",",
            "\\\"b\\\":\\\"2\\\",",
            "\\\"c\\\":\\\"3\\\"}",
        );
        let v = lenient_parse_object(body).expect("three-key rescue must parse");
        assert_eq!(v["a"], "line1\nline2");
        assert_eq!(v["b"], "2");
        assert_eq!(v["c"], "3");
    }

    #[test]
    fn gemma4_write_with_bare_path_key_and_literal_newlines() {
        // Worst-case observed: the second key is BARE (no quotes) AND
        // the first value contains literal newlines. Strict fails on
        // controls; quote_bare_keys alone leaves controls; the
        // escape-controls path applied after quoting is what should
        // rescue this.
        let body = "{\"content\":\"line 1\nline 2\" ,path:\"wordcount.py\"}";
        let v = lenient_parse_object(body).expect("bare-key + literal-newline body must parse");
        assert_eq!(v["path"], "wordcount.py");
        assert_eq!(v["content"], "line 1\nline 2");
    }

    #[test]
    fn unquoted_key_and_value() {
        let v = lenient_parse_object(
            r#"{prompt: find docs about job search activity in my drive}"#,
        ).unwrap();
        assert_eq!(
            v["prompt"],
            "find docs about job search activity in my drive",
        );
    }

    #[test]
    fn quoted_key_unquoted_value() {
        let v = lenient_parse_object(r#"{"prompt": find docs about job search}"#).unwrap();
        assert_eq!(v["prompt"], "find docs about job search");
    }

    #[test]
    fn bare_key_numeric_value_kept_as_number() {
        let v = lenient_parse_object(r#"{limit: 10}"#).unwrap();
        assert_eq!(v["limit"], 10);
    }

    #[test]
    fn bare_key_array_value_parsed() {
        let v = lenient_parse_object(r#"{all_of: ["a","b"]}"#).unwrap();
        assert_eq!(v["all_of"][0], "a");
        assert_eq!(v["all_of"][1], "b");
    }

    #[test]
    fn ambiguous_multi_bare_value_refused() {
        // Two keys, both with bare-string values — where does `foo bar`
        // end and `b` begin? Refuse.
        let r = lenient_parse_object(r#"{a: foo bar, b: baz qux}"#);
        assert!(r.is_none(), "ambiguous multi-bare should not parse");
    }

    #[test]
    fn quote_bare_keys_skips_inside_strings() {
        // `key:` looks like a key but it's inside a string literal —
        // must not get re-quoted.
        let out = quote_bare_keys(r#"{prompt: "this has key: in it"}"#);
        assert_eq!(out, r#"{"prompt": "this has key: in it"}"#);
    }

    #[test]
    fn quote_bare_keys_preserves_escaped_quotes() {
        let out = quote_bare_keys(r#"{x: "a\"b", y: 1}"#);
        assert_eq!(out, r#"{"x": "a\"b", "y": 1}"#);
    }

    #[test]
    fn empty_object_passthrough() {
        let v = lenient_parse_object(r#"{}"#).unwrap();
        assert!(v.is_object());
        assert!(v.as_object().unwrap().is_empty());
    }

    #[test]
    fn equals_separator_with_array_value() {
        // Gemma 4 mixed-separator body observed in the wild — `any_of`
        // uses `=` and the date keys use `:`. quote_bare_keys must
        // normalise both forms.
        let v = lenient_parse_object(
            r#"{any_of=["employment","job","hiring"],from:"2026-05-09",to:"2026-05-16"}"#,
        ).unwrap();
        assert_eq!(v["any_of"][0], "employment");
        assert_eq!(v["any_of"][2], "hiring");
        assert_eq!(v["from"], "2026-05-09");
        assert_eq!(v["to"], "2026-05-16");
    }

    #[test]
    fn equals_inside_string_value_preserved() {
        // The `=` rewrite must not touch separators that live inside
        // existing string literals.
        let out = quote_bare_keys(r#"{prompt: "a=b"}"#);
        assert_eq!(out, r#"{"prompt": "a=b"}"#);
    }

    #[test]
    fn equals_separator_quotes_bare_key() {
        let out = quote_bare_keys(r#"{x=1, y=2}"#);
        assert_eq!(out, r#"{"x":1, "y":2}"#);
    }
}
