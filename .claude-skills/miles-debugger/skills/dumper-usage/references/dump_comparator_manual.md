# Comparator

## 1. CLI Reference

```bash
python -m sglang.srt.debug_utils.comparator [OPTIONS]
```

**Required:**


| Flag              | Type  | Description                     |
| ----------------- | ----- | ------------------------------- |
| `--baseline-path` | `str` | Path to baseline dump directory |
| `--target-path`   | `str` | Path to target dump directory   |


If a given path contains no `.pt` files but has exactly one subdirectory, the comparator automatically descends into it.

**Filtering & selection:**


| Flag                      | Type     | Default   | Description                                                                            |
| ------------------------- | -------- | --------- | -------------------------------------------------------------------------------------- |
| `--start-step`            | `int`    | `0`       | Include steps >= this value (target side only)                                         |
| `--end-step`              | `int`    | `1000000` | Include steps <= this value (target side only)                                         |
| `--filter`                | `str`    | `None`    | Regex pattern to filter filenames (target side only; include only matches)             |
| `--grouping-skip-keys`    | `str...` | —         | Metadata keys to skip when grouping files into bundles                                 |
| `--allow-skipped-pattern` | `str`    | `".*"`    | Regex (`fullmatch`) for tensor names allowed to be skipped without affecting exit code |
| `--allow-failed-pattern`  | `str`    | `None`    | Regex (`fullmatch`) for tensor names allowed to fail without affecting exit code       |


**Comparison:**


| Flag               | Type    | Default | Description                                                                           |
| ------------------ | ------- | ------- | ------------------------------------------------------------------------------------- |
| `--diff-threshold` | `float` | `1e-3`  | Relative difference threshold for pass/fail                                           |
| `--token-aligner`  | `str`   | `None`  | Token alignment mode:`smart` or `concat_steps`                                        |
| `--tokenizer`      | `str`   | `None`  | Tokenizer path for decoding input_ids. Auto-discovered from dump metadata if not set. |


**Dims override:**


| Flag                       | Type               | Description                                       |
| -------------------------- | ------------------ | ------------------------------------------------- |
| `--override-dims`          | `str` (repeatable) | Override dims for both sides:`"name:dims_string"` |
| `--override-baseline-dims` | `str` (repeatable) | Override dims for baseline only                   |
| `--override-target-dims`   | `str` (repeatable) | Override dims for target only                     |
| `--override-config`        | `str`              | Path to YAML override config file                 |


**Preset:**


| Flag       | Type  | Choices                                | Description                                              |
| ---------- | ----- | -------------------------------------- | -------------------------------------------------------- |
| `--preset` | `str` | `raw`, `sglang_dev`, `sglang_megatron` | Predefined configuration (see[Section 2](#2-presets)) |


**Output & report:**


| Flag              | Type  | Default                                 | Description                                                  |
| ----------------- | ----- | --------------------------------------- | ------------------------------------------------------------ |
| `--output-format` | `str` | `"text"`                                | `text` (human-readable) or `json` (JSONL)                    |
| `--verbosity`     | `str` | `"normal"`                              | `minimal`, `normal`, or `verbose`                            |
| `--report-path`   | `str` | `<target-path>/comparator_report.jsonl` | JSONL report output path. Pass empty string `""` to disable. |


**Visualization:**


| Flag                    | Type  | Default                  | Description                                               |
| ----------------------- | ----- | ------------------------ | --------------------------------------------------------- |
| `--viz-bundle-details`  | flag  | `False`                  | Generate PNG heatmaps for each tensor pair                |
| `--viz-output-dir`      | `str` | `"/tmp/comparator_viz/"` | Output directory for visualization PNGs                   |
| `--visualize-per-token` | `str` | `None`                   | Output path for per-token relative difference heatmap PNG |


**Exit code:** `0` = all passed (at least one), non-zero = failures, errors, disallowed skips, or zero passed tensors.

## 2. Presets


| Preset                 | Expands To                                                    | Typical Use Case                                |
| ---------------------- | ------------------------------------------------------------- | ----------------------------------------------- |
| `raw`                  | `--grouping-skip-keys` (empty)                                | No cross-rank grouping                          |
| `sglang_dev` (default) | `--grouping-skip-keys rank`                                   | Same framework, different hardware/config       |
| `sglang_megatron`      | `--grouping-skip-keys rank step --token-aligner concat_steps` | Cross-framework comparison (SGLang vs Megatron) |


Automatic behaviors:

- If neither `--preset` nor `--grouping-skip-keys` is specified, `sglang_dev` is applied automatically.
- `dump_index` and `filename` are always skipped in grouping regardless of preset.

**What `grouping-skip-keys` means:** The comparator groups dump files into bundles — one bundle per logical tensor. A bundle is formed by matching metadata keys. Keys listed in `grouping-skip-keys` are excluded from this matching, so files that differ only in those keys are grouped together. For example, skipping `rank` causes files from different ranks with the same `(step, name, ...)` to form a single bundle, which the unsharder then reassembles.

## 3. Alignment Pipeline

For each tensor bundle, the comparator runs a multi-stage alignment pipeline before element-wise comparison:

```
Raw .pt files (multiple ranks, multiple steps)
    │
    ▼
[Unsharder]     per-step: multiple ranks → single full tensor
    │
    ▼
[Reorderer]     per-step: fix special orderings (e.g. zigzag → natural)
    │
    ▼
[Token Aligner] cross-step: multiple step tensors → single aligned tensor
    │
    ▼
[Axis Aligner]  cross-side: unify dimension order and fuse/squeeze
    │
    ▼
Element-wise comparison
```

Before alignment begins, **DP filtering** is applied: when a tensor has a `dp_group_alias` (from the `# dp:=...` annotation), the comparator selects only the files belonging to a single non-empty DP rank, discarding the rest.

### Unsharder

The unsharder reassembles sharded tensors back into full tensors based on the `dims` annotation's parallel modifiers. Each axis is processed independently, starting from the innermost (rightmost) modifier. Replicated axes are always processed before sharded axes.


| Strategy       | Condition                                            | Operation                                                                                              |
| -------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Concat**     | Default sharded axis (e.g.`h[tp]`)                   | `torch.cat` all rank shards along the annotated dimension                                              |
| **Reduce-sum** | `partial` qualifier (e.g. `h[tp:partial]`)           | `torch.stack(tensors).sum(dim=0)`                                                                      |
| **Pick**       | `replicated` declaration (e.g. `# tp:replicated`)    | Take rank 0's tensor; verify all ranks match (atol=1e-6)                                               |
| **THD Concat** | CP axis on token dimension with `seq_lens` available | Split each rank's data by per-sequence lengths, interleave across ranks per-sequence, then concatenate |


When multiple axes are present (e.g. both TP and CP):

- For each target axis, group tensors by their coordinates on all *other* axes. For example, when unsharding CP with TP=2, tensors sharing the same TP rank are grouped together.
- Within each group, sort by the target axis rank and combine.
- Axes are processed one at a time, from innermost to outermost.

### Reorderer

The reorderer undoes CP zigzag interleaving. With CP=N, the sequence is divided into 2N chunks and interleaved for load balancing:

```
Natural order:  [A0] [A1] [B0] [B1]     (CP=2, 4 chunks)
Zigzag order:   [A0] [B0] [A1] [B1]     (interleaved)
```

The reorderer restores natural order. Two modes:

- **Sequence dimension zigzag**: Reorder the full tensor's sequence dimension directly. Used when per-sequence lengths are unavailable.
- **Token dimension zigzag (THD)**: Split the tensor by `seq_lens` into per-sequence segments, reorder each segment independently, then concatenate. Used for packed token data where each sequence has variable length.

### Token Aligner

Handles alignment across multiple dump steps (e.g. inference with multiple forward passes vs training with a single forward pass).

`**concat_steps` mode** (simple):

- Concatenate all steps' tensors along the token dimension, sorted by step number.
- Truncate to `min(total_baseline_tokens, total_target_tokens)`.
- Suitable for BS=1 scenarios without sequence-level matching.

`**smart` mode** (precise):

- Requires auxiliary tensors (`input_ids`, `positions`, `seq_lens`, etc.) to be present in the dumps.
- Auto-detects the framework via discriminating fields (e.g. `seq_lens` → SGLang, `cu_seqlens_q` → Megatron).
- Detects token layout: `T` (flat token stream, e.g. SGLang THD format) or `BS` (batch × sequence, e.g. Megatron BSHD format). `BS` tensors are flattened to `T` before matching.
- Sequence matching algorithm (two passes):
    1. **Exact match**: Pair sequences whose `input_ids` are identical across baseline and target.
    2. **Prefix match**: If no exact match, find the best pairing where one sequence's `input_ids` is a prefix of the other's (handles generation scenarios where lengths differ).
- After matching, extract the common prefix tokens from each matched pair and reassemble into an aligned tensor.

### Axis Aligner

Handles cases where both sides have the same semantic dimension names but in different order or with different fusing.

Example:

- Baseline dims: `h d` (two separate dimensions)
- Target dims: `(h*d)` (fused into one dimension)
- Resolution: Apply `rearrange("h d -> (h d)")` to the baseline to match the target's shape.

Matching rules:

1. Extract semantic dimension names from both sides (expand fused groups, ignore squeeze dims `1`).
2. Verify the name sets are identical.
3. Build a canonical layout (using one side as reference) and generate einops `rearrange` patterns for each side.

## 4. Override Config

YAML format for `--override-config`:

```yaml
overrides:
  - match: "regex_pattern"     # regex matched against tensor name (first match wins)
    dims: "b s h[tp] d"        # dims string to apply
    side: both                  # both | baseline | target
```

CLI equivalents:

```bash
--override-dims "name:dims"             # both sides
--override-baseline-dims "name:dims"    # baseline only
--override-target-dims "name:dims"      # target only
```

## 5. Output & Report

**Text format (terminal):**

Three verbosity levels:

- `minimal`: One line per tensor — name and pass/fail only.
- `normal`: Compact lifecycle display — shape transformations and diff values.
- `verbose`: Full detail per tensor — statistics, sample values, replicated-axis check results.

A summary line is printed at the end: total / passed / failed / skipped / errored.

**JSON format (JSONL):**

Each line is a JSON object, discriminated by the `type` field:


| Type                    | Description                                                                                                                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config`                | Configuration snapshot (first record)                                                                                                                                                    |
| `comparison_tensor`     | Tensor comparison result (main body). Includes: shape, statistics (mean/std/min/max/percentiles), diff (rel_diff/max_abs_diff/passed), raw_bundle_info, replicated_checks, errors/infos. |
| `comparison_non_tensor` | Non-tensor value comparison (equality check)                                                                                                                                             |
| `comparison_skip`       | Skipped tensor with reason (e.g.`baseline_load_failed`)                                                                                                                                  |
| `comparison_error`      | Exception during comparison                                                                                                                                                              |
| `rank_info`             | Rank topology information                                                                                                                                                                |
| `input_ids`             | Decoded input_ids for each side (when tokenizer is available)                                                                                                                            |
| `log`                   | Informational log messages                                                                                                                                                               |
| `summary`               | Aggregate statistics (last record)                                                                                                                                                       |


**Report file:** By default written to `<target-path>/comparator_report.jsonl`. Pass `--report-path ""` to disable.

## 6. Visualization

- `--viz-bundle-details`: Generate a PNG for each tensor pair (heatmap + statistics histogram). Output to `--viz-output-dir`.
- `--visualize-per-token <path>`: Generate a per-token relative difference heatmap PNG (y-axis = tensor name, x-axis = token position).