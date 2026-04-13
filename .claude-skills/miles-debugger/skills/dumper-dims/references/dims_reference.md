# References to dumper `dims` annotation

TODO: after tests are ok, link to the tests as well as a reference

We firstly describe the grammars and definitions, and then provide a comprehensive list as examples.

## 1 Descriptions

### 1.1 Syntax

Full BNF grammar:

```
dims_string    ::= dims_part [ "#" comment_part ]
dims_part      ::= dim_token { " " dim_token }
dim_token      ::= "1" | plain_dim | fused_dim
plain_dim      ::= identifier [ "[" modifiers "]" ]
fused_dim      ::= "(" sub_dims ")" [ "[" modifiers "]" ]
sub_dims       ::= identifier "*" identifier { "*" identifier }
identifier     ::= [a-zA-Z_]\w*
modifiers      ::= modifier { "," modifier }
modifier       ::= axis [ ":" qualifiers ]
qualifiers     ::= qualifier { "+" qualifier }
qualifier      ::= "zigzag" | "natural" | "partial" | "sharded"
axis           ::= "tp" | "cp" | "ep" | "sp" | "recompute_pseudo"
comment_part   ::= { declaration }
declaration    ::= "dp:=" identifier | axis ":replicated"
```

### 1.2 Dimension Names


| Type      | Description                                                                                               | Examples                            |
| --------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| Plain     | Arbitrarily named dimension, can be single or multi characters                                            | `h`, `d`, `vocab_size`, `num_heads` |
| Special   | `t` := num_tokens, `b` := batch_size, `s` := seq_lens (following the "thd" and "bshd" naming in Megatron) | -                                   |
| Fused     | Multiple semantic dims stored as one physical dim                                                         | `(num_heads*head_dim)`              |
| Singleton | Size-1 dimension, squeezed by the comparator                                                              | `1`                                 |


### 1.3 Modifiers

**Parallel Axes**

The axis name must match the dumped parallel information. For example, SGLang dumps `tp_rank` / `tp_size`, thus a potential axis is `tp`.

**Reduction Qualifiers**


| Qualifier           | Description                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------- |
| `partial`           | Partial reduction — the tensor holds a partial sum before all-reduce. The unsharder applies reduce-sum. |
| `sharded` (default) | No-op, purely for readability (having an axis already implies sharded)                                  |


**Ordering Qualifiers**


| Qualifier           | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `zigzag`            | Zigzag interleaved ordering (commonly used with CP)                         |
| `natural` (default) | Natural sequential ordering (default if no ordering qualifier is specified) |


**Combinations**

They can be combined, such as:

```
h[cp:zigzag+partial,sp]    # CP-sharded with zigzag ordering and partial reduction, and then SP-sharded
```

#### 1.4 Comment Section

Declarations after `#`, separated by spaces:


| Declaration         | Description                                        | Example           |
| ------------------- | -------------------------------------------------- | ----------------- |
| `dp:=<name>`        | Set the DP group alias for this tensor             | `# dp:=moe_dp`    |
| `<axis>:replicated` | Declare that this axis is replicated (not sharded) | `# tp:replicated` |


An axis cannot be declared as both sharded (in the dims part) and replicated (in the comment section).

Multiple replicated declarations can be combined:

```
b s h d # tp:replicated ep:replicated
```

## 2 Examples

Below are examples for several typical cases in SGLang and Megatron. Other code paths and configurations will have different annotations.

### 2.1 SGLang MoE Model

#### 2.1.1 TP Attention

Standard TP mode, attention heads sharded across TP:


| Tensor                      | Dims                                        | Description                                                                                                                                                                                                       |
| --------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Layer input                 | `t h # tp:replicated`                       | After all-reduce, all TP ranks hold the same full hidden state                                                                                                                                                    |
| Attn after-Q                | `t h[tp]` (or `t (num_heads*head_dim)[tp]`) | SGLang fuses head dims into one, and TP shards the fused dimension; the latter form is needed when matching dimensions with Megatron (which is unfused); the same `h` in different dumps can mean different sizes |
| Attn pre-O                  | `t h[tp]`                                   | Each TP rank owns a shard of attention heads                                                                                                                                                                      |
| Attn output                 | `t h[tp:partial]`                           | Each TP rank computed partial result; needs reduce-sum to reconstruct                                                                                                                                             |
| MoE input (assuming TP MoE) | `t h # tp:replicated`                       | Also replicated across TP                                                                                                                                                                                         |


#### 2.1.2 DP Attention

DP-attention mode, attention tensors are fully data parallel:


| Tensor                      | Dims                                 | Description                                                                                                                                                                                                             |
| --------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Layer input                 | `t h # dp:=attn_dp`                  | DP-attention distributes tokens across DP ranks;`dp:=attn_dp` tells comparator to use the `attn_dp` group for data parallel (this can be omitted because the default group name, `dp`, is equivalent to `attn_dp` here) |
| Attn after-Q                | `t h # dp:=attn_dp`                  | -                                                                                                                                                                                                                       |
| Attn pre-O                  | `t h # dp:=attn_dp`                  | -                                                                                                                                                                                                                       |
| Attn output                 | `t h # dp:=attn_dp`                  | -                                                                                                                                                                                                                       |
| MoE input (assuming TP MoE) | `t h # moe_tp:replicated dp:=moe_dp` | In DP Attention mode, MoE has different TP/DP group than global (attention) groups, thus need explicit annotations                                                                                                      |


#### 2.1.3 TP MoE

【TODO: wrong, not write yet!!! and not yet fully implemented EP】

MoE experts sharded across TP, no expert parallelism:


| Tensor              | Dims                         | Description                                                                        |
| ------------------- | ---------------------------- | ---------------------------------------------------------------------------------- |
| MoE sublayer inputs | `t h # moe_tp:replicated`    | -                                                                                  |
| Router logits       | `t topk # moe_tp:replicated` | Gating is computed on the full hidden state, so all ranks produce identical logits |
| After Gate/Up GEMM  | `TODO`                       | TODO                                                                               |
| Expert output       | `t h[tp:partial]`            | Same as MLP output — experts are TP-sharded                                        |


#### 2.1.4 EP MoE via AllGather

【TODO: wrong, not write yet!!! and not yet fully implemented EP】

TODO: ep to be implemented


| Tensor              | Dims                                       | Description                                                       |
| ------------------- | ------------------------------------------ | ----------------------------------------------------------------- |
| MoE sublayer inputs | `t h # ep:replicated`                      | -                                                                 |
| Router logits       | `t topk # tp:replicated moe_tp:replicated` | Gating computed on full hidden state; identical across all axes   |
| After Gate/Up GEMM  | `TODO`                                     | TODO                                                              |
| Expert output       | `t h[tp:partial] # moe_tp:replicated`      | Experts are TP-sharded (partial sum) but replicated across MoE-TP |


#### 2.1.5 EP MoE via DeepEP

【TODO: wrong, not write yet!!! and not yet fully implemented EP】

TODO: ep to be implemented


| Tensor              | Dims                                       | Description                                                       |
| ------------------- | ------------------------------------------ | ----------------------------------------------------------------- |
| MoE sublayer inputs | `t[ep] h`                                  | TODO: "ep"                                                        |
| Router logits       | `t topk # tp:replicated moe_tp:replicated` | Gating computed on full hidden state; identical across all axes   |
| After Gate/Up GEMM  | `TODO`                                     | TODO                                                              |
| Expert output       | `t h[tp:partial] # moe_tp:replicated`      | Experts are TP-sharded (partial sum) but replicated across MoE-TP |


### 2.2 Megatron MoE Model

#### 2.2.1 THD vs BSHD Format

Megatron supports both layouts. The only difference in dims is the first two dimensions:


| Layout              | Decoder-layer dims    | Attention dims                             |
| ------------------- | --------------------- | ------------------------------------------ |
| THD (packed tokens) | `t[cp:zigzag,sp] 1 h` | `t[cp:zigzag,sp] num_heads[tp] head_dim`   |
| BSHD (batch × seq)  | `s[cp:zigzag,sp] b h` | `s[cp:zigzag,sp] b num_heads[tp] head_dim` |


THD uses `t` (packed token stream) with a singleton `1` for batch; BSHD uses `s` (sequence) with explicit `b` for batch. The comparator's axis aligner handles mapping between them.

#### 2.2.2 Full BSHD Example

Megatron side with CP+SP zigzag and multiple replicated axes. Shorthand: `REPL` = `tp:replicated ep:replicated etp:replicated`.


| Tensor                   | Dims                                                                                           | Description                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Layer input              | `s[cp:zigzag,sp] b h # REPL`                                                                   | Seq dim is CP-zigzag + SP sharded; hidden state is replicated across TP/EP/ETP |
| Attn output              | `s[cp:zigzag,sp] b h # REPL`                                                                   | Same as layer input — post-all-reduce, only seq dim is sharded                 |
| Pre-MLP residual         | `s[cp:zigzag,sp] b h # REPL`                                                                   | Same layout                                                                    |
| Pre-MLP layernorm output | `s[cp:zigzag,sp] b h # REPL`                                                                   | Same layout                                                                    |
| MLP output               | `s[cp:zigzag,sp] b h # REPL`                                                                   | Same layout                                                                    |
| Attn Q                   | `s[cp:zigzag,sp] b num_heads[tp] head_dim # ep:replicated etp:replicated`                      | Megatron keeps head dims separate;`num_heads` is TP-sharded                    |
| Attn K                   | `s[cp:zigzag,sp] b num_kv_heads[tp] head_dim # ep:replicated etp:replicated`                   | KV uses fewer heads (GQA); same TP sharding                                    |
| Attn V                   | `s[cp:zigzag,sp] b num_kv_heads[tp] head_dim # ep:replicated etp:replicated`                   | Same as Attn K                                                                 |
| Attn pre-O-proj          | `s[cp:zigzag,sp] b (num_heads*head_dim)[tp] # ep:replicated etp:replicated`                    | After attention core; fused back to single dim, still TP-sharded               |
| Router logits            | `s[cp:zigzag,sp] b num_experts # REPL`                                                         | Gating on full hidden state; identical across all axes                         |
| Top-K IDs                | `t[cp:zigzag,sp] topk # REPL`                                                                  | Always token-major (even in BSHD); replicated                                  |
| Expert output            | `t h[tp:partial] # ep:replicated etp:replicated moe_tp:replicated cp:replicated sp:replicated` | Experts are TP-sharded (partial sum); replicated on all other axes             |

