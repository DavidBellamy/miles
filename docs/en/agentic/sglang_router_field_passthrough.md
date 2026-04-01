# SGLang Router: Request Field Passthrough Behavior

## Problem

The SGLang Rust Router (sgl-model-gateway) has **two different routing backends** that handle
request fields differently. The default backend **silently drops unknown request fields**,
which breaks TITO session tracking that relies on `return_meta_info` and `return_prompt_token_ids`.

## Two Router Backends

### HTTP Router (`--backend sglang`, default)

The default "regular" backend. Request flow:

```
Client JSON → serde deserialize into ChatCompletionRequest struct → re-serialize to JSON → forward to worker
```

**Unknown fields are dropped.** Serde's default behavior discards any JSON fields not defined in
the Rust `ChatCompletionRequest` struct. Fields like `return_meta_info`, `return_prompt_token_ids`,
and `return_routed_experts` are all lost because the Rust struct (defined in the external
`openai-protocol` crate) does not include them.

Relevant code: `sgl-model-gateway/src/routers/http/router.rs:546`:
```rust
self.client
    .post(format!("{}{}", worker_url, route))
    .json(typed_req)  // ← re-serializes the typed struct, losing unknown fields
```

### OpenAI Router (`--backend openai`)

An alternative backend designed for proxying to external OpenAI-compatible endpoints.
Request flow:

```
Client JSON → serde deserialize into ChatCompletionRequest → convert to serde_json::Value → forward to worker
```

**Unknown fields are preserved.** By converting the typed struct back to a generic
`serde_json::Value` before forwarding, all original fields survive the round-trip.

Relevant code: `sgl-model-gateway/src/routers/openai/router.rs:510-576`:
```rust
let mut payload = to_value(body)?;  // ← preserves all fields as JSON Value
// ...
let mut req = client.post(&url).json(&*payload);  // ← sends full Value
```

## Impact on TITO / Session Server

The session server (`miles/rollout/session/sessions.py`) injects these fields into
every chat completion request before proxying to the inference backend:

```python
request_body["logprobs"] = True
request_body["return_prompt_token_ids"] = True   # ← dropped by HTTP router
request_body["return_meta_info"] = True           # ← dropped by HTTP router
```

When the HTTP router drops `return_meta_info`, the Python sglang server defaults it to
`False` and omits `meta_info` from the response. The session server then fails with:

```
RuntimeError: meta_info and output_token_logprobs must be in choice (requires logprobs=True)
```

## Backend Selection

The backend is controlled by the `--backend` CLI flag passed to the sglang router binary:

| Value | Router Used | Unknown Fields | Use Case |
|-------|-------------|----------------|----------|
| `sglang` (default) | HTTP Router | **Dropped** | Internal sglang workers |
| `openai` | OpenAI Router | **Preserved** | External OpenAI-compatible endpoints |
| `vllm` | HTTP Router | **Dropped** | vLLM workers |
| `trtllm` | HTTP Router | **Dropped** | TensorRT-LLM workers |
| `anthropic` | HTTP Router | **Dropped** | Anthropic-compatible endpoints |

## Solutions

### Option 1: Use `--backend openai` (workaround)

Pass `--backend openai` when starting the sglang router so that requests are proxied
as generic JSON values instead of typed structs.

### Option 2: Upstream fix — add `#[serde(flatten)]` catch-all

Add a catch-all field to the Rust `ChatCompletionRequest` struct:

```rust
pub struct ChatCompletionRequest {
    pub model: String,
    // ... existing fields ...

    #[serde(flatten)]
    pub extra_fields: HashMap<String, serde_json::Value>,
}
```

This preserves all unknown fields through the deserialize → re-serialize round-trip.

### Option 3: Bypass the Rust Router for session requests

Have the session server proxy directly to sglang worker engines, bypassing the Rust
Router entirely. This is what commit `939b161d` implemented, but it introduces complexity
around worker URL discovery and load balancing.

## Key Files

- `sgl-model-gateway/src/main.rs:869-884` — Backend selection logic
- `sgl-model-gateway/src/routers/http/router.rs:485-547` — HTTP router typed forwarding
- `sgl-model-gateway/src/routers/openai/router.rs:470-576` — OpenAI router Value forwarding
- `sgl-model-gateway/src/server.rs:184-193` — Endpoint handler deserialization
- `sgl-model-gateway/src/routers/factory.rs:23-62` — Router factory instantiation
- `miles/rollout/session/sessions.py:84-86` — Fields injected by session server
