# Source Patcher

## 1. YAML Config Format

```yaml
patches:
  - target: "module.Class.method"     # fully qualified function name (required)
    preamble: |                        # injected at function body start (optional)
      import ...
    edits:                             # list of edits, applied sequentially (required)
      - match: |                       # source text to find (required)
          original_code
        append: |                      # add after matched text (one of three modes)
          new_code
      - match: "..."
        prepend: "..."                 # add before matched text (one of three modes)
      - match: "..."
        replacement: "..."             # replace matched text (one of three modes)
```

## 2. Matching Rules

- Both the source code and the `match` text are split into lines and stripped (leading/trailing whitespace removed) before comparison.
- Matching is **exact** (stripped lines must be equal). No regex or substring matching.
- The match must be **unique**: the patcher raises `PatchApplicationError` if the text is found 0 times or more than once.
- Multiple edits are applied sequentially — later edits see the result of earlier edits.

## 3. Edit Modes


| Mode          | Behavior                              | Original Text |
| ------------- | ------------------------------------- | ------------- |
| `append`      | Add new lines after the matched text  | Preserved     |
| `prepend`     | Add new lines before the matched text | Preserved     |
| `replacement` | Replace matched text with new text    | Removed       |


The three modes are mutually exclusive — each edit uses at most one. If none is specified, the matched text is deleted.

Indentation is automatically aligned to the indentation level of the first matched line.

## 4. Preamble

The `preamble` is injected at the start of the function body (immediately after the signature line, before any code including docstrings). Indentation is automatically aligned to the function body.

Typical use: injecting import statements.

## 5. Auto Import Injection

When invoked through `dumper.apply_source_patches()`, the dumper automatically prepends `from sglang.srt.debug_utils.dumper import dumper` to every patch's preamble. There is no need to include this import in the YAML.

## 6. API

```python
# High-level: apply from YAML string
from sglang.srt.debug_utils.source_patcher import apply_patches_from_config
states = apply_patches_from_config(yaml_content, extra_imports=["from ... import ..."])

# Via dumper (reads DUMPER_SOURCE_PATCHER_CONFIG env var)
dumper.apply_source_patches()

# Low-level: context manager with automatic rollback
from sglang.srt.debug_utils.source_patcher import CodePatcher, PatchSpec
with CodePatcher(patches=[PatchSpec(...)]) as patcher:
    ...  # function is patched here
# original code is restored on exit
```

