//! Render the GGUF's embedded Jinja chat template directly with
//! `minijinja`, instead of going through llama.cpp's hand-rolled
//! `llama_chat_apply_template` (which doesn't recognize Gemma 4's
//! template — it returns `FfiError(-1)`).
//!
//! Why a real Jinja engine matters: the Gemma 4 chat template
//! contains a `{%- for tool in tools %}` block + `format_function_declaration`
//! macro that, when fed `tools=[{type:"function", function:{name, description, parameters}}, …]`,
//! renders a `<|tool>declaration:NAME{...schema...}<tool|>` block
//! the model is *trained* to read. With that block present the
//! model emits real `<|tool_call>call:NAME{ARGS}<tool_call|>` calls.
//! Without it (our previous prose-only approach) the model just
//! emits a `<|tool_call>call<tool_call|>` null placeholder and
//! either stops or fabricates.
//!
//! Implementation notes:
//! - Template + (messages, tools) → string. Tokenization is the
//!   caller's job.
//! - `enable_thinking=true` so the template doesn't inject the
//!   `<|channel>thought\n<channel|>` thinking-off sentinel — the
//!   deep model is the reasoning model.
//! - One `minijinja::Environment` per call. The template's roughly
//!   16KB and parses in single-digit ms; LazyLock-caching a parsed
//!   environment is a follow-up if profiling shows it matters.

use model_api_proto::ToolDef;
use minijinja::{context, Environment, Value};
use serde_json::json;

/// A single chat message handed to the template. Roles match
/// OpenAI's chat-completion spec (`system` / `user` / `assistant` /
/// `tool`) since that's what every Jinja-style template (Gemma 4,
/// Qwen, Llama 3.1+, …) iterates.
#[derive(Clone, Debug)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

impl ChatMsg {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}

/// Render `template` with the model-template's expected variable
/// names. `tools` becomes the JSON array the template iterates
/// (each entry is `{type:"function", function:{name, description, parameters}}`).
///
/// Returns the fully-rendered prompt string ready to tokenize.
pub fn render_chat(
    template: &str,
    messages: &[ChatMsg],
    tools: &[ToolDef],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<String, String> {
    let msgs_json: Vec<Value> = messages
        .iter()
        .map(|m| {
            Value::from_serialize(json!({
                "role": m.role,
                "content": m.content,
            }))
        })
        .collect();

    let tools_json: Vec<Value> = tools
        .iter()
        .map(|t| Value::from_serialize(tool_to_openai_function(t)))
        .collect();

    let mut env = Environment::new();
    // The chat template is sourced from the GGUF (`model.chat_template`),
    // not from disk — name is arbitrary, only used in error messages.
    env.add_template("chat", template)
        .map_err(|e| format!("jinja parse: {e}"))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| format!("jinja get_template: {e}"))?;

    tmpl.render(context! {
        messages => msgs_json,
        tools => tools_json,
        add_generation_prompt => add_generation_prompt,
        enable_thinking => enable_thinking,
    })
    .map_err(|e| format!("jinja render: {e}"))
}

/// Map `ToolDef` → OpenAI function-call spec JSON
/// (`{type:"function", function:{name, description, parameters: {type:"object", properties, required}}}`),
/// which is what every modern chat template's `{% for tool in tools %}`
/// loop expects.
fn tool_to_openai_function(def: &ToolDef) -> serde_json::Value {
    let mut properties = serde_json::Map::new();
    let mut required: Vec<serde_json::Value> = Vec::new();
    for p in &def.parameters {
        let mut prop = serde_json::Map::new();
        prop.insert(
            "type".into(),
            serde_json::Value::String(json_schema_type(&p.param_type)),
        );
        if !p.description.is_empty() {
            prop.insert(
                "description".into(),
                serde_json::Value::String(p.description.clone()),
            );
        }
        properties.insert(p.name.clone(), serde_json::Value::Object(prop));
        if p.required {
            required.push(serde_json::Value::String(p.name.clone()));
        }
    }
    let parameters = json!({
        "type": "object",
        "properties": properties,
        "required": required,
    });
    json!({
        "type": "function",
        "function": {
            "name": def.name,
            "description": def.description,
            "parameters": parameters,
        }
    })
}

/// Coerce the loose `ToolDef::param_type` strings (which can
/// be `"string"` / `"str"` / `"STRING"` / `"integer"` / `"int"` /
/// `"boolean"` / `"bool"` / etc., depending on whether the def came
/// from a JSON file or an MCP server's `tools/list`) into a canonical
/// JSON-schema type word. Unknown types pass through unchanged.
fn json_schema_type(raw: &str) -> String {
    match raw.to_ascii_lowercase().as_str() {
        "str" | "string" => "string".into(),
        "int" | "integer" => "integer".into(),
        "num" | "number" | "float" => "number".into(),
        "bool" | "boolean" => "boolean".into(),
        "arr" | "array" | "list" => "array".into(),
        "obj" | "object" | "dict" | "map" => "object".into(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_api_proto::{ToolDef, ToolParam};

    fn remember_def() -> ToolDef {
        ToolDef {
            name: "remember".into(),
            description: "Store a memory.".into(),
            parameters: vec![
                ToolParam {
                    name: "key".into(),
                    param_type: "string".into(),
                    required: true,
                    description: "Identifier".into(),
                },
                ToolParam {
                    name: "value".into(),
                    param_type: "string".into(),
                    required: true,
                    description: "Value to remember".into(),
                },
            ],
        }
    }

    #[test]
    fn render_minimal_template() {
        // Sanity check: minijinja handles {% for %} + member access
        // over our tools schema (the same patterns the Gemma 4
        // template uses).
        let tmpl = "{% for t in tools %}{{ t.function.name }}={{ t.function.parameters.required | length }};{% endfor %}";
        let out = render_chat(tmpl, &[], &[remember_def()], true, true).unwrap();
        assert_eq!(out, "remember=2;");
    }

    #[test]
    fn render_messages_with_roles() {
        let tmpl = "{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}{% if add_generation_prompt %}<go>{% endif %}";
        let out = render_chat(
            tmpl,
            &[ChatMsg::system("sys"), ChatMsg::user("hi")],
            &[],
            true,
            false,
        )
        .unwrap();
        assert_eq!(out, "[system:sys][user:hi]<go>");
    }
}
