---
name: debug-sft
description: Use when debugging Supervised Fine-Tuning (SFT) issues in Miles, including SFT training producing gibberish output, checkpoint conversion errors for SFT, VLM/multimodal SFT issues, tokenization problems, chat template application errors, double forward pass in SFT, or SFT-specific data pipeline issues. Also trigger on reward always returning 0, deepscaler reward bugs, or tau-bench failures.
---

# Debug SFT (Supervised Fine-Tuning) Issues

SFT in Miles has unique failure modes related to checkpoint conversion, tokenization, and the training-inference pipeline.

## Quick Decision Tree

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Good loss but gibberish output | Checkpoint conversion or tokenizer mismatch | Verify weight round-trip, check special tokens |
| `ValueError: Unknown parameter name` | Model type not in converter | Add parameter mapping for model architecture |
| `partition_stride != 1 not supported` | VLM model has unsupported parameter layout | Needs custom converter support |
| Double FLOPs during SFT | Redundant logprob forward pass | Remove extra `_compute_log_prob()` call |
| `jinja2.UndefinedError` in tokenization | `rollout_max_prompt_len` is None | Set explicit value or handle None |
| Reward always 0 with instruct model | Reward parser expects thinking tags | Fix parser to handle marker-less responses |
| `beta_fast=1.0` missing | Megatron YaRN defaults wrong | Add `--beta-fast 1.0` to launch args |

## Common SFT Bugs

### Bug 1: Good Training Loss but Gibberish Output (Miles #650)

**The #1 SFT failure mode**. Training completes, loss converges, but the model outputs nonsense.

**Possible causes** (check in order):

1. **Tokenizer mismatch**: Training and inference use different tokenizer configs
   ```python
   # Verify:
   from transformers import AutoTokenizer
   train_tok = AutoTokenizer.from_pretrained(train_path)
   infer_tok = AutoTokenizer.from_pretrained(infer_path)
   test = "Hello, how are you?"
   assert train_tok.encode(test) == infer_tok.encode(test), "Tokenizer mismatch!"
   assert train_tok.all_special_tokens == infer_tok.all_special_tokens
   ```

2. **Chat template not applied or applied wrong**:
   ```python
   # Check if chat template is needed
   messages = [{"role": "user", "content": "Hello"}]
   formatted = train_tok.apply_chat_template(messages, tokenize=False)
   print(formatted)  # Should include proper system prompt, turn markers, etc.
   ```

3. **Embedding/output layer weight tying broken**:
   ```python
   # Some models tie input_embedding and output_layer weights
   # If conversion separates them, the output layer may have wrong weights
   sd = torch.load('model.pt')
   emb = sd.get('model.embed_tokens.weight')
   lm_head = sd.get('lm_head.weight')
   if emb is not None and lm_head is not None:
       print(f"Tied: {torch.equal(emb, lm_head)}")
   ```

4. **Special token IDs shifted**: BOS/EOS/PAD token IDs may differ between model versions

### Bug 2: VLM Checkpoint Conversion Failure (Miles #634)

**Symptom**: `ValueError: Unknown parameter name: module.module.vision_model.patch_embed.proj.weight`

**Root cause**: The `megatron_to_hf/` converter only maps language model parameters. Vision tower (ViT) parameters need separate handling.

**Affected files**: `miles/backends/megatron_utils/megatron_to_hf/qwen2.py` (and model-specific converters)

**Workaround**: Manually merge vision tower weights from original HF checkpoint:
```python
# Load converted language model weights
hf_state_dict = torch.load('converted_model.pt')
# Load original vision tower weights
original_sd = torch.load('original_model/pytorch_model.bin')
vision_keys = [k for k in original_sd if 'vision_model' in k]
for k in vision_keys:
    hf_state_dict[k] = original_sd[k]
torch.save(hf_state_dict, 'merged_model.pt')
```

### Bug 3: Double Forward Pass (Miles #333)

**Symptom**: SFT training takes 2x expected FLOPs.

**Root cause**: In FSDP actor, redundant `self._compute_log_prob("actor", packed_batches)` call during SFT (not needed since SFT doesn't use PPO-style log-prob comparison).

**Fix**: Skip the extra forward pass in SFT mode.

### Bug 4: Reward Always Returns 0 (Miles #330)

**Symptom**: DeepScaler rule-based reward returns 0 for all samples with non-thinking models.

**Root cause**: `get_deepscaler_rule_based_reward()` returns 0 if neither `</think>` nor `###Response` markers found in the response.

**Fix**: When markers are absent, use `model_solution = response` instead of `return 0`.

### Bug 5: Tokenization Crash with None Lengths (Miles #349)

**Symptom**: `jinja2.exceptions.UndefinedError: int object has no element 0`

**Root cause**: `rollout_max_prompt_len` defaults to `None`, causing downstream tokenization operations to fail on indexing.

**Fix**: Set explicit defaults or handle None in tokenization pipeline.

### Bug 6: Wrong YaRN Parameters (Miles #335)

**Symptom**: Doubled training-inference difference with Kimi-K2 model.

**Root cause**: Megatron defaults `beta_fast=32` but Kimi-K2 needs `beta_fast=1.0`. No CLI arg exists for this in Megatron-Core.

**Workaround**: When using MBridge (Megatron Bridge), it reads correctly from HF config. Direct Megatron args path needs manual override.

## SFT Data Pipeline

### Chat Template Application

```bash
--apply-chat-template                    # Apply chat template during data processing
--apply-chat-template-kwargs '{"enable_thinking": false}'  # For instruct models
```

**Important**: Different models need different chat templates. Verify:
```python
# Check which template the model uses
print(tokenizer.chat_template)
# Should show Jinja2 template with role markers, system prompt, etc.
```

### SFT Loss Computation

SFT uses standard cross-entropy loss on the response tokens only:
```
Input:  [system_prompt | user_turn | assistant_response]
Loss:    ----masked----   --masked-   ^^^loss computed^^^
```

The loss mask must correctly exclude prompt tokens. If the mask is wrong, the model learns to copy the prompt rather than generate responses.

## VLM/Multimodal SFT

### Supported Models
- Qwen3-VL-*: Partially supported (vision tower conversion incomplete)
- Other VLMs: May need custom converter support

### Key Challenges
1. Vision tower weights are not in the Megatron converter
2. Image preprocessing pipeline must match between training and inference
3. Vision-language connector weights need correct mapping
4. `partition_stride != 1` error indicates multi-modal parameters with non-standard sharding

## Debugging SFT Training

### Step 1: Verify Training Data

```python
# Check a sample from the training data
from datasets import load_dataset
ds = load_dataset('parquet', data_files='train.parquet')
sample = ds['train'][0]
print(sample.keys())
print(sample['messages'][:2])  # First user-assistant turn
```

### Step 2: Verify Loss Is Computing on Correct Tokens

```python
# Add to training loop:
dumper.dump('loss_mask', loss_mask, dims='t')
dumper.dump('tokens', input_ids, dims='t')
# Check: loss_mask should be 1 only for response tokens
```

### Step 3: Compare Pre/Post Fine-Tuning Outputs

```python
# Generate with base model:
base_output = base_model.generate(prompt)
# Generate with fine-tuned model:
ft_output = ft_model.generate(prompt)
print(f"Base: {base_output}")
print(f"Fine-tuned: {ft_output}")
# If ft_output is gibberish but base_output is coherent → conversion bug
```

## Related Skills

- `/debug-weight-sync`: For checkpoint conversion issues
- `/debug-logprob`: For SFT log-prob computation
- `/debug-precision`: For numerical issues during SFT training
