//! Multi-token-prediction (speculative decode) integration with PR22673.
//!
//! Status: API exploration only. Wiring is gated on three pieces of
//! work that aren't done yet:
//!
//! 1. **wrapper.h additions in llama-cpp-2.** mtp-clean's MTP API lives
//!    in `deps/llama.cpp/common/speculative.h`, not `include/llama.h`.
//!    bindgen via the current `wrapper.h` does not see it. To expose
//!    `common_speculative_init`, `common_speculative_begin`,
//!    `common_speculative_draft`, `common_speculative_accept`, etc.,
//!    add `#include "llama.cpp/common/speculative.h"` to `wrapper.h`
//!    (or to a new `wrapper_speculative.h`) on the
//!    `main-llamacpp-gemma4` branch of `deps/llama-cpp-2`.
//!
//! 2. **Draft-context construction.** The MTP path uses a second
//!    `LlamaContext` of type `LLAMA_CONTEXT_TYPE_MTP` (see
//!    `llama.h:203`). llama-cpp-2's `LlamaContextParams` does not
//!    surface `ctx_type` yet; need to thread it through, or call the
//!    sys API directly.
//!
//! 3. **Inner-loop integration.** The `Session` step/push_token API
//!    is single-token. MTP drafts `n` tokens, the verifier accepts a
//!    prefix, and the rest are dropped. The right integration point
//!    is a new `Session::step_mtp(spec) -> (LlamaToken, n_accepted)`
//!    that wraps `common_speculative_draft` + per-token verify.
//!
//! Config knobs already in `lib::ThinkerConfig`:
//!   `draft_gguf_path: Option<String>` — when `Some`, a second
//!     GGUF is loaded as the draft model. Currently unused at
//!     runtime; `lib::spawn` ignores it.
//!   `mtp_draft_n: u32` — how many tokens to propose per step.
//!
//! Reference: `am17an/mtp-clean` commits a55493b..e7b4848.
