//! Gemma chat-template assembly via llama-cpp-2.
//!
//! The model's embedded jinja template is the source of truth — we
//! don't reimplement Gemma 3-4's role tags in Rust. `build_chat_prompt`
//! takes a structured message list and returns the prefilled prompt
//! string ready to tokenize.
//!
//! Today's `lib::spawn` consumes `ThinkerRequest::input` as a
//! pre-built prompt string (the existing path through
//! `thinker_impl::build_prompt` + ChatMLFormat). To exercise Gemma's
//! native template, callers will plumb raw messages through to this
//! module instead — that's the next stage of the thinker_impl
//! integration.

use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};

/// One message in a chat turn.
pub struct Message<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

/// Apply the GGUF's embedded chat template to a message list.
/// `add_assistant_prefix = true` appends the assistant role opening
/// (typical for completion requests; off for evaluation against an
/// existing assistant turn).
pub fn build_chat_prompt(
    model: &LlamaModel,
    messages: &[Message<'_>],
    add_assistant_prefix: bool,
) -> Result<String, String> {
    let template = model
        .chat_template(None)
        .map_err(|e| format!("chat_template: {e:?}"))?;

    let chat: Result<Vec<LlamaChatMessage>, _> = messages
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.to_string(), m.content.to_string()))
        .collect();
    let chat = chat.map_err(|e| format!("LlamaChatMessage::new: {e:?}"))?;

    model
        .apply_chat_template(&template, &chat, add_assistant_prefix)
        .map_err(|e| format!("apply_chat_template: {e:?}"))
}

/// Convenience: assemble a (system, user) two-turn prompt and apply
/// Gemma's template. Mirrors how `thinker_impl::escalate` builds a
/// single-turn request for the GPU path.
///
/// When `tools` is non-empty AND the model's embedded chat template
/// has the Gemma 4 markers (`<|turn>`), the full Jinja template is
/// rendered via `minijinja` (see `jinja_chat::render_chat`) so the
/// `{% for tool in tools %}` block emits the native
/// `<|tool>declaration:NAME{schema}<tool|>` definition section the
/// model is trained to read. Without that section the model just
/// emits null `<|tool_call>call<tool_call|>` placeholders and
/// fabricates answers.
///
/// When `tools` is empty the hand-rolled `build_gemma4_chat` is
/// faster and avoids the Jinja parse cost.
pub fn build_simple_chat(
    model: &LlamaModel,
    system_prompt: Option<&str>,
    user_text: &str,
    tools: &[model_api_proto::ToolDef],
) -> Result<String, String> {
    let mut messages: Vec<Message> = Vec::with_capacity(2);
    if let Some(sys) = system_prompt {
        messages.push(Message { role: "system", content: sys });
    }
    messages.push(Message { role: "user", content: user_text });

    // Gemma 4 ships a custom jinja chat template (uses `<|turn>` /
    // `<turn|>` markers, `<|channel>thought\n<channel|>` for the
    // thinking-on/off switch, plus a full tools-definition block)
    // that none of llama.cpp's built-in templates recognize —
    // `apply_chat_template` returns FfiError(-1) for it. Sniff the
    // GGUF's embedded template for the Gemma 4 marker and route
    // through the matching path; fall back to the built-in dispatch
    // for non-Gemma-4 models.
    if let Ok(tmpl) = model.chat_template(None) {
        if let Ok(tmpl_str) = tmpl.to_str() {
            if tmpl_str.contains("<|turn>") {
                if tools.is_empty() {
                    // No tools to advertise → cheap hand-rolled wrap.
                    return Ok(build_gemma4_chat(system_prompt, user_text));
                }
                // Tools present → full Jinja render so the template's
                // `format_function_declaration` macro emits real
                // tool definitions.
                let jinja_msgs: Vec<crate::jinja_chat::ChatMsg> = messages
                    .iter()
                    .map(|m| crate::jinja_chat::ChatMsg {
                        role: m.role.to_string(),
                        content: m.content.to_string(),
                    })
                    .collect();
                return crate::jinja_chat::render_chat(
                    tmpl_str,
                    &jinja_msgs,
                    tools,
                    /* add_generation_prompt = */ true,
                    /* enable_thinking = */ true,
                );
            }
        }
    }
    build_chat_prompt(model, &messages, true)
}

/// Hand-rolled Gemma 4 chat template wrap. Format derived from the
/// 26B-A4B-it GGUF's embedded jinja:
///   `<|turn>{role}\n{content}<turn|>\n` per message,
///   then `<|turn>model\n` to open the assistant turn for generation.
/// Thinking stays ON (model's default) — we don't emit the
/// `<|channel>thought\n<channel|>` thinking-off suppression marker
/// because the deep thinker is the reasoning model.
fn build_gemma4_chat(system_prompt: Option<&str>, user_text: &str) -> String {
    let mut s = String::with_capacity(user_text.len() + 128);
    if let Some(sys) = system_prompt {
        if !sys.is_empty() {
            s.push_str("<|turn>system\n");
            s.push_str(sys);
            s.push_str("<turn|>\n");
        }
    }
    s.push_str("<|turn>user\n");
    s.push_str(user_text);
    s.push_str("<turn|>\n");
    s.push_str("<|turn>model\n");
    s
}
