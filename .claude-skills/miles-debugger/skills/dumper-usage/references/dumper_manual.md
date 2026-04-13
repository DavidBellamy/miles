# Dumper

## 1. Core API

The dumper is a global singleton imported as:

```python
from sglang.srt.debug_utils.dumper import dumper
```

`**dumper.dump(name, value, save=True, dims=None, dims_grad=None, **kwargs)**`

Dump a single value (tensor, scalar, or arbitrary object).


| Parameter   | Type   | Description                                                                                                                              |
| ----------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `name`      | `str`  | Identifier used in the filename and for comparator matching. Supports hierarchical naming with `__` (e.g. `"layer__attention__output"`). |
| `value`     | `Any`  | The tensor or value to dump.                                                                                                             |
| `save`      | `bool` | If `False`, skips both file output and `capture_output()` capture. Default `True`.                                                       |
| `dims`      | `str   | None`                                                                                                                                    |
| `dims_grad` | `str   | None`                                                                                                                                    |
| `**kwargs`  |        | Custom tags (e.g.`layer_id=0`, `phase="prefill"`). These become part of the metadata and filename.                                       |


`**dumper.step()**`

Increment the step counter. Call this at the end of each iteration. SGLang's training/inference loop calls this automatically.

`**dumper.dump_model(model, name_prefix="param", save=True, **kwargs)**`

Dump all named parameters of a `torch.nn.Module`. Each parameter is saved as a separate dump with name `{name_prefix}__{param_name}` (e.g. `param__layers.0.weight`). Whether values and gradients are dumped is controlled by `enable_model_value` and `enable_model_grad` respectively.

`**dumper.dump_dict(name_prefix, data, save=True, **kwargs)**`

Dump all items in a dict (or all non-callable attributes of an object). Each item is saved as `{name_prefix}_{key}`.

`**dumper.capture_output()**`

Context manager that captures all dumps made within the `with` block into an in-memory dict (values are cloned). Works regardless of the `enable_output_file` setting.

```python
with dumper.capture_output() as captured:
    dumper.dump("x", tensor_x)
    dumper.dump("y", tensor_y)

# captured == {"x": {"value": ..., "meta": {...}}, "y": {"value": ..., "meta": {...}}}
```

## 2. Context & Filtering

`**dumper.set_ctx(**kwargs)**`

Set global context variables that are automatically included in every subsequent dump's metadata and available in filter expressions. Pass `None` to clear a variable:

```python
dumper.set_ctx(layer_id=5, phase="decode")
dumper.set_ctx(layer_id=None)  # clear layer_id
```

`**@dumper.ctx(...)` decorator**

Set context for the duration of a function call, then clear it on exit. Two forms:

```python
# Lambda extractor: extract context from `self`
@dumper.ctx(lambda self: dict(layer_id=self.layer_id))
def forward(self, hidden_states, ...): ...

# Static kwargs
@dumper.ctx(phase="decode")
def decode_step(self, x): ...
```

**Filter expressions**

The `filter` config accepts a Python expression evaluated against each dump's tags dict.

Available variables: `name`, `recompute_status`, any custom `**kwargs` passed to `dump()`, and any context variables set via `set_ctx()` (e.g. `layer_id`). Note that `step`, `rank`, and `dump_index` are **not** available in filter expressions. Missing variables resolve to `None`.

Built-in functions: `search(pattern, string)` and `match(pattern, string)` (i.e. `re.search` / `re.match`).

Examples:

```python
# Only layers 0-3
"layer_id in [0, 1, 2, 3]"

# Only dumps whose name contains "attention"
"search('attention', name)"

# Combined
"layer_id in range(0, 4) and not search('grad', name)"
```

## 3. Configuration

All configuration fields and their environment variables:


| Field                   | Env Var                        | Type | Default         | Description                                                                               |
| ----------------------- | ------------------------------ | ---- | --------------- | ----------------------------------------------------------------------------------------- |
| `enable`                | `DUMPER_ENABLE`                | bool | `False`         | Master enable switch                                                                      |
| `filter`                | `DUMPER_FILTER`                | str  | `None`          | Filter expression                                                                         |
| `dir`                   | `DUMPER_DIR`                   | str  | `"/tmp/dumper"` | Output directory                                                                          |
| `enable_output_file`    | `DUMPER_ENABLE_OUTPUT_FILE`    | bool | `True`          | Write `.pt` files                                                                         |
| `enable_output_console` | `DUMPER_ENABLE_OUTPUT_CONSOLE` | bool | `True`          | Print to console                                                                          |
| `enable_value`          | `DUMPER_ENABLE_VALUE`          | bool | `True`          | Dump tensor values                                                                        |
| `enable_grad`           | `DUMPER_ENABLE_GRAD`           | bool | `False`         | Register grad hooks to capture gradients                                                  |
| `enable_model_value`    | `DUMPER_ENABLE_MODEL_VALUE`    | bool | `False`         | `dump_model()` captures parameter values                                                  |
| `enable_model_grad`     | `DUMPER_ENABLE_MODEL_GRAD`     | bool | `False`         | `dump_model()` captures parameter gradients                                               |
| `exp_name`              | `DUMPER_EXP_NAME`              | str  | `None`          | Experiment name. Auto-generated as `dump_YYYYMMDD_HHMMSS_MMMrrr` (ms + random) if `None`. |
| `cleanup_previous`      | `DUMPER_CLEANUP_PREVIOUS`      | bool | `False`         | Delete old `dump_*` directories before first write                                        |
| `collective_timeout`    | `DUMPER_COLLECTIVE_TIMEOUT`    | int  | `60`            | Timeout (seconds) for rank synchronization                                                |
| `server_port`           | `DUMPER_SERVER_PORT`           | str  | `"-1"`          | HTTP control port.`-1` = disabled, `"reuse"` = reuse the framework's existing port.       |
| `non_intrusive_mode`    | `DUMPER_NON_INTRUSIVE_MODE`    | str  | `"core"`        | Non-intrusive hook mode:`"core"`, `"all"`, or `"off"`                                     |
| `source_patcher_config` | `DUMPER_SOURCE_PATCHER_CONFIG` | str  | `None`          | Path to source patcher YAML config                                                        |


Programmatic configuration:


| Method                                             | Description                                               |
| -------------------------------------------------- | --------------------------------------------------------- |
| `DumperConfig.from_env()`                          | Construct config from environment variables               |
| `DumperConfig.from_kv_pairs(["enable=true", ...])` | Construct from key=value strings                          |
| `dumper.configure(**kwargs)`                       | Update config at runtime (partial update)                 |
| `dumper.configure_default(**kwargs)`               | Set defaults without overriding existing values           |
| `dumper.get_state()`                               | Returns `{"config": ..., "dump_index": ..., "step": ...}` |
| `dumper.reset()`                                   | Clear step counter and remove non-intrusive hooks         |


## 4. HTTP Control

Requires `DUMPER_SERVER_PORT` to be set to a valid port (e.g. `30000`) or `"reuse"`.


| Endpoint            | Method | Request Body                             | Description                         |
| ------------------- | ------ | ---------------------------------------- | ----------------------------------- |
| `/dumper/get_state` | POST   | `{}`                                     | Returns config, dump_index, step    |
| `/dumper/configure` | POST   | `{"enable": true, "filter": "...", ...}` | Partial config update               |
| `/dumper/reset`     | POST   | `{}`                                     | Reset state and non-intrusive hooks |


In multi-rank scenarios, requests are automatically broadcast to all ranks via ZMQ. The response is a JSON array with one element per rank.

## 5. Non-intrusive Mode

Activated by `dumper.register_non_intrusive_dumper(model)`, which registers PyTorch forward hooks on all modules.


| Mode     | Behavior                                                                                                                           |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `"core"` | Only dump framework core fields (see table below). Names are clean with no prefix.                                                 |
| `"all"`  | Dump all module inputs/outputs. Names are prefixed with `non_intrusive`__ (e.g. `non_intrusive__model.layers.0.self_attn.output`). |
| `"off"`  | Disabled                                                                                                                           |


Core fields by framework:


| Framework | Core Fields                                                                |
| --------- | -------------------------------------------------------------------------- |
| SGLang    | `input_ids`, `positions`, `seq_lens`, `req_pool_indices`, `rids`           |
| Megatron  | `input_ids`, `position_ids`, `cu_seqlens_q`, `cu_seqlens_kv`, `qkv_format` |


`layer_id` is automatically detected and injected into context for modules whose name matches `layers.\d+`:

1. First, framework plugin detection is attempted (SGLang: `module.layer_id`, Megatron: `module.layer_number`).
2. If the plugin returns nothing, falls back to extracting the integer from the module name regex match.

## 6. Output Format

Directory structure:

```
<dir>/<exp_name>/
  ├── step=0___rank=0___dump_index=1___name=hidden_states___layer_id=0.pt
  ├── step=0___rank=0___dump_index=2___name=attn_output___layer_id=0.pt
  └── ...
```

Filename rules:

- Tags are separated by `___` (triple underscore).
- Fixed tag order: `step`, `rank`, `dump_index`, `name`, followed by custom tags.

Each `.pt` file contains:

```python
{
  "value": <tensor or other value>,
  "meta": {
    # Always present
    "step": int,
    "rank": int,
    "dump_index": int,
    "name": str,
    "recompute_status": "disabled" | "original" | "recompute",

    # Static metadata (constant within a single run)
    "world_rank": int,
    "world_size": int,
    "<framework>_parallel_info": {"tp_rank": 0, "tp_size": 4, "pp_rank": 0, "pp_size": 2, ...},
    "tokenizer_path": str | None,

    # Optional
    "dims": str,           # if dims was passed to dump()
    "dims_grad": str,      # if dims_grad was passed to dump()
    "<custom_key>": value,  # from **kwargs
  }
}
```
