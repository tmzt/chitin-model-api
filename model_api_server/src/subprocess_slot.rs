//! Subprocess-backed `SlotHandle`: per-request, spawns
//! `llama-completion` (or any binary with the same CLI shape),
//! reads its stdout chunk-by-chunk, streams those as
//! `StreamChunk`s through the `StreamSink`, then returns the
//! full text as the `InferenceResponse`.
//!
//! Why a subprocess and not direct FFI to libllama: this lets us
//! reuse the cross-built llama-completion + libggml-vulkan.so we
//! already ship on the Pixel (stage 5) without forcing
//! llama-cpp-2 to cross-compile through the workspace's Rust dep
//! tree. Same llama.cpp build either way; the wire to the model
//! just runs through stdout pipes instead of an in-process call.
//!
//! Per-request startup cost: ~5 s on CPU for a 0.5B model (model
//! load); inference itself runs at the model's normal speed.
//! Acceptable for the demo wire. Stage 11+ can swap to a
//! long-lived backend slot once Vulkan PowerVR shaders compile
//! reliably (Q5_0/Q8_0 matmul currently fail on this driver).

use async_trait::async_trait;
use model_api_proto::{InferenceInput, InferenceResponse, ProgressEvent, StreamChunk};
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::slot::{DiscardSink, SlotHandle, SlotRequest, SlotResponse, StreamSink};

pub struct SubprocessSlot {
    /// Path to llama-completion (or llama-cli — same arg shape).
    pub llama_bin: PathBuf,
    /// Path to the .gguf model file.
    pub model: PathBuf,
    /// `LD_LIBRARY_PATH` exported to the child so it finds
    /// libggml-base / libggml-cpu / libggml-vulkan / libllama / libomp.
    pub lib_dir: PathBuf,
    /// Layers to offload to GPU. `99` = all (Vulkan); `0` = CPU
    /// only. On the Pixel we currently default to 0 — the PowerVR
    /// driver fails to compile some matmul shader variants.
    pub n_gpu_layers: i32,
    pub model_name: String,
}

impl SubprocessSlot {
    pub fn new(
        llama_bin: PathBuf,
        model: PathBuf,
        lib_dir: PathBuf,
        n_gpu_layers: i32,
    ) -> Self {
        let model_name = model
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "subprocess".to_string());
        Self { llama_bin, model, lib_dir, n_gpu_layers, model_name }
    }
}

#[async_trait]
impl SlotHandle for SubprocessSlot {
    fn model_name(&self) -> &str { &self.model_name }
    fn gpu_memory_mb(&self) -> Option<u32> { None }

    async fn run(&self, req: SlotRequest) -> Result<SlotResponse, String> {
        self.run_stream(req, &DiscardSink).await
    }

    async fn run_stream(
        &self,
        req: SlotRequest,
        sink: &dyn StreamSink,
    ) -> Result<SlotResponse, String> {
        let prompt = match &req.req.input {
            InferenceInput::Text(t) => t.clone(),
            _ => return Err("subprocess slot: non-text input not supported".into()),
        };
        let max_tokens = req.req.max_tokens.max(1) as u32;

        sink.on_progress(ProgressEvent {
            phase: "queued".into(), tool: None, detail: None,
        });

        let mut cmd = Command::new(&self.llama_bin);
        cmd.env("LD_LIBRARY_PATH", &self.lib_dir)
            .arg("-m").arg(&self.model)
            .arg("-p").arg(&prompt)
            .arg("-n").arg(max_tokens.to_string())
            .arg("-ngl").arg(self.n_gpu_layers.to_string())
            .arg("--no-warmup")
            .arg("-no-cnv")
            .arg("--no-display-prompt")
            .arg("--temp").arg("0.0")  // deterministic for now
            .arg("--seed").arg("42")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        sink.on_progress(ProgressEvent {
            phase: "gen_start".into(), tool: None,
            detail: Some(format!("ngl={}", self.n_gpu_layers)),
        });

        log::info!("[subprocess-slot] spawn: {} -m {} -n {max_tokens}",
            self.llama_bin.display(), self.model.display());
        let mut child = cmd.spawn().map_err(|e| format!("spawn: {e}"))?;
        let mut stdout = child.stdout.take().ok_or("stdout pipe missing")?;

        // Read stdout in fixed-size chunks; flush to the sink
        // every ~16 chars (or whatever lands per read). This is
        // good enough for streaming-feel; the wire chunks land at
        // ~one-per-token cadence either way.
        let mut full_text = String::new();
        let mut buf = [0u8; 256];
        let mut acc = String::new();
        loop {
            let n = match stdout.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    log::warn!("[subprocess-slot] read: {e}");
                    break;
                }
            };
            match std::str::from_utf8(&buf[..n]) {
                Ok(s) => {
                    full_text.push_str(s);
                    acc.push_str(s);
                }
                Err(e) => {
                    // Partial UTF-8 at chunk boundary — fall back
                    // to lossy; pixel doesn't care.
                    let lossy = String::from_utf8_lossy(&buf[..n]);
                    full_text.push_str(&lossy);
                    acc.push_str(&lossy);
                    log::debug!("[subprocess-slot] utf8 boundary ({e}) — lossy decode");
                }
            }
            if acc.len() >= 16 {
                sink.on_chunk(StreamChunk {
                    delta_text: std::mem::take(&mut acc),
                    finish_reason: None,
                    phase: Some("text".into()),
                });
            }
        }
        if !acc.is_empty() {
            sink.on_chunk(StreamChunk {
                delta_text: acc,
                finish_reason: None,
                phase: Some("text".into()),
            });
        }
        sink.on_chunk(StreamChunk {
            delta_text: String::new(),
            finish_reason: Some("stop".into()),
            phase: Some("text".into()),
        });

        let _ = child.wait();
        sink.on_progress(ProgressEvent {
            phase: "gen_done".into(), tool: None,
            detail: Some(format!("{} chars", full_text.len())),
        });

        Ok(SlotResponse(InferenceResponse {
            text: full_text,
            session_id: None,
            raw_text: None,
            injections: Vec::new(),
            tool_calls: Vec::new(),
            replacement: None,
        }))
    }
}
