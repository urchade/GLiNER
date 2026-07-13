# GLiNER-Robust Repo Map (Phase 0, Agent A)

Scope: `feat/conformal-prediction` branch, vanilla upstream GLiNER checkout (upstream/main tip
`f33bace`). All paths are relative to repo root
`/Users/aliiii/Desktop/projects/GLINER/GLiNER-Robust`. Every claim below is backed by a direct
read of the cited file/lines in this checkout — nothing here is inferred from prior knowledge of
GLiNER's public releases.

---

## 1. Package layout

`gliner/` top-level modules (each with line counts from `wc -l` at time of writing):

- `gliner/__init__.py` — public API surface: exports `GLiNER`, `GLiNERConfig`,
  `InferencePackingConfig`, `PackedBatch`, `pack_requests`, `unpack_spans`. `__version__ = "0.2.27"`.
- `gliner/model.py` (5005 lines) — the whole model-class hierarchy (`BaseGLiNER` and every
  concrete variant) plus the dispatching `GLiNER` meta-class, `from_pretrained`, inference/decode
  orchestration (`inference`, `run_batch`, `decode_batch`, `predict_entities`,
  `batch_predict_entities`), evaluation entrypoints, and prompt-embedding compression utilities.
- `gliner/config.py` (386 lines) — `BaseGLiNERConfig` and all per-architecture config subclasses;
  registers them into `transformers`' `CONFIG_MAPPING`.
- `gliner/modeling/` — the actual `nn.Module` graph: `base.py` (model forward passes and losses,
  2718 lines), `encoder.py` (997 lines, text/backbone encoding + `get_representations`),
  `decoder.py` (440 lines, the label-generation decoder used by "SpanDecoder"/"TokenDecoder"
  *model* variants — NOT the same thing as `gliner/decoding/decoder.py`, see §5), `span_rep.py`
  (759 lines, span representation layers, e.g. `SpanRepLayer`, `markerV0` mode), `scorers.py`
  (81 lines, the token-level `Scorer` module), `outputs.py` (108 lines, `GLiNERBaseOutput` /
  `GLiNERDecoderOutput` / `GLiNERRelexOutput` dataclasses — this is what `forward()` returns),
  `loss_functions.py`, `utils.py`, `multitask/` (relation/triple extraction layers).
- `gliner/decoding/` — post-forward-pass decoding logic: `decoder.py` (1915 lines, sigmoid +
  threshold + greedy overlap resolution — see §4/§5), `utils.py` (19 lines, `has_overlapping` /
  `has_overlapping_nested`), `trie/` (constrained generation trie for generative label decoding).
- `gliner/data_processing/` — tokenization, span-index construction, batch collation
  (`processor.py`, `tokenizer.py`, `collator.py`, `utils.py`).
- `gliner/evaluation/` — `evaluate_ner.py` (CoNLL-style dataset loading/scripted eval),
  `evaluator.py` (`BaseNEREvaluator`, `BaseRelexEvaluator`, precision/recall/F1), `utils.py`.
- `gliner/onnx/model.py` — ONNX Runtime wrapper classes mirroring the PyTorch model hierarchy.
- `gliner/serve/` — a Ray Serve-based production serving layer (dynamic batching, memory
  calibration, PolyLoRA adapter serving) — unrelated to core inference correctness.
- `gliner/training/trainer.py` — HF-`Trainer`-based training loop (`Trainer`, `TrainingArguments`).
- `gliner/multitask/` — higher-level task wrappers (classification, QA, summarization, open
  extraction, relation extraction) built on top of `GLiNER`; currently commented out of
  `gliner/__init__.py` (lines 12–14) so not part of the public import surface today.
- `gliner/infer_packing.py` — request packing for inference (`InferencePackingConfig`,
  `pack_requests`, `unpack_spans`).
- `gliner/utils.py` — misc helpers (e.g. `is_module_available`).

Note on scope vs. expectations: this checkout is considerably more elaborate than the "classic"
GLiNER public release (fp16/bf16 variant downloads, `torch.compile`, int8 quantization,
`low_cpu_mem_usage` meta-device loading, prompt-embedding compression/distillation, inference
packing, decoder-based generative label variants, relation-extraction "relex" variants, a Ray
Serve layer). Treat every fact below as specific to *this* checkout, not to GLiNER in general.

---

## 2. The `GLiNER` class and `from_pretrained` flow

File: `gliner/model.py`.

### Class hierarchy

```
BaseGLiNER(ABC, nn.Module, PyTorchModelHubMixin)          # model.py:112
├── BaseEncoderGLiNER(BaseGLiNER)                          # model.py:1800
│   ├── BaseBiEncoderGLiNER(BaseEncoderGLiNER)              # model.py:2671
│   │   ├── BiEncoderSpanGLiNER(BaseBiEncoderGLiNER)        # model.py:3064
│   │   └── BiEncoderTokenGLiNER(BaseBiEncoderGLiNER)       # model.py:3106
│   ├── UniEncoderSpanGLiNER(BaseEncoderGLiNER)             # model.py:2940
│   ├── UniEncoderTokenGLiNER(BaseEncoderGLiNER)            # model.py:3006
│   ├── UniEncoderSpanDecoderGLiNER(BaseEncoderGLiNER)      # model.py:3144  (generative label decoder)
│   │   └── UniEncoderTokenDecoderGLiNER(...)               # model.py:3573
│   └── UniEncoderSpanRelexGLiNER(BaseEncoderGLiNER)        # model.py:3588  (joint NER + relation extraction)
│       └── UniEncoderTokenRelexGLiNER(...)                 # model.py:4450

GLiNER(nn.Module, PyTorchModelHubMixin)                    # model.py:4533  (dispatcher, NOT a subclass of BaseGLiNER)
```

`GLiNER` (model.py:4533) is a **self-replacing dispatcher**, not a real base class member. Its
`__init__` (model.py:4568) loads/normalizes the config, calls the static method
`_get_gliner_class(config)` (model.py:4607), instantiates that concrete class, then does
`self.__class__ = type(new_instance); self.__dict__ = new_instance.__dict__` (model.py:4604-4605)
— i.e. `GLiNER(...)` mutates itself into whichever concrete subclass matches. Dispatch logic
(model.py:4609-4641) branches on `config.relations_layer`, `config.labels_decoder`,
`config.labels_encoder`, and `config.span_mode == "token_level"` to pick among the 8 leaf classes
listed above.

**No poly-encoder exists.** Grepped case-insensitively for "poly" across `gliner/`: every hit is
`PolyLoRA` (an unrelated LoRA-adapter serving feature in `gliner/serve/`, e.g.
`gliner/serve/config.py:61-70`, `gliner/serve/server.py:131-235`). There is no poly-encoder
*architecture* (the retrieval-style shared-context/candidate-embedding encoder concept) anywhere
in this codebase. The only two encoder families are **uni-encoder** (single shared text encoder,
entity-label prompts prepended into the same sequence, e.g. `UniEncoderSpanGLiNER`) and
**bi-encoder** (separate text encoder and label encoder, `BaseBiEncoderGLiNER`, model.py:2671).

### `from_pretrained` flow

Two `from_pretrained` classmethods exist:

- `BaseGLiNER.from_pretrained` — model.py:1037-1799ish. This is where the actual loading logic
  lives: resolves `variant`/`dtype` (model.py:1128-1154), downloads or locates the model dir
  (`_download_model`, model.py:1157-1168), loads `gliner_config.json` via `_load_config`
  (model.py:1171-1181), loads the tokenizer (`_load_tokenizer`, model.py:1184-1189), resolves the
  weights file (`_resolve_model_file`) and either builds normally or (if
  `low_cpu_mem_usage=True`) builds on `torch.device("meta")` and swaps in tensors via
  `load_state_dict(assign=True)` (model.py:1200-1216ff).
- `GLiNER.from_pretrained` — model.py:4644 (a classmethod on the dispatcher). Reads
  `gliner_config.json` to determine the concrete subclass first, then delegates to that
  subclass's own `from_pretrained` (inherited from `BaseGLiNER`).

Config file convention: `gliner_config.json` inside the model directory (model.py:1171-1173);
`FileNotFoundError` is raised if absent.

---

## 3. Where span logits/scores are produced — exact shapes per encoder path

All forward passes return a `GLiNERBaseOutput` (or subclass) dataclass, defined in
`gliner/modeling/outputs.py:8-40`. Key fields: `logits`, `span_idx`, `span_mask`, `span_logits`.

### 3a. Uni-encoder, **span** mode — `UniEncoderSpanModel`

File: `gliner/modeling/base.py:383-488` (class at 383, `forward` at 414).

```python
prompts_embedding = self.prompt_rep_layer(prompts_embedding)          # base.py:473
scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)  # base.py:474
```

- `span_rep`: `(B, L, K, D)` — produced by `self.span_rep_layer(words_embedding, span_idx)`
  (base.py:463), where `L` = number of word positions, `K` = `config.max_width` (max span width),
  `D` = `hidden_size`.
- `prompts_embedding`: `(B, C, D)`, `C` = number of entity-type prompts.
- **`scores` (= `logits` in the returned `GLiNERBaseOutput`) has shape `(B, L, K, C)`** — raw,
  real-valued, pre-sigmoid. Confirmed by the docstring at base.py:508 ("Predicted scores of shape
  (B, L, K, C)") and the `loss()` method's own unpacking `BS, _, _, CL = scores.shape`
  (base.py:528).

### 3b. Bi-encoder, **span** mode — `BiEncoderSpanModel`

File: `gliner/modeling/base.py:889-1005` (class at 889, `forward` at 917).

Identical einsum, same shape:

```python
scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)  # base.py:991
```

**`(B, L, K, C)`**, same semantics as 3a. The only difference vs. the uni-encoder path is that
`prompts_embedding`/`prompts_embedding_mask` come from a *separate* label encoder
(`labels_embeds`/`labels_input_ids`/`labels_attention_mask` params, base.py:921-923) rather than
being extracted from the same sequence as the text.

### 3c. Uni-encoder, **token** mode — `UniEncoderTokenModel`

File: `gliner/modeling/base.py:560-763` (class at 560, `forward` at 609).

```python
# Shape: (batch_size, seq_len, num_classes, 3), 3 - start, end, inside
scores = self.scorer(words_embedding, prompts_embedding)   # base.py:671-672
```

**`scores` (= `logits`) has shape `(B, W, C, 3)`** where `W` = number of words, `C` = number of
entity types, and the trailing dim of size 3 is `[start, end, inside]` compatibility scores —
produced by `Scorer.forward` (`gliner/modeling/scorers.py:45-81`), whose own docstring
(scorers.py:55) and code (`nn.Linear(hidden_size * 4, 3)` at scorers.py:42) confirm the `3`.

If `config.represent_spans` is truthy (base.py:582, 674), the model *additionally* derives
span-level logits from the token-level scores via `get_span_representations`
(base.py:590-607) and:

```python
span_logits = torch.einsum("BND,BCD->BNC", span_rep, prompts_embedding)  # base.py:678
```

giving a **second** score tensor of shape `(B, N, C)` (`N` = number of extracted candidate spans,
variable/data-dependent), returned as `output.span_logits` alongside `output.span_idx`
(`(B, N, 2)`) and `output.span_mask` (`(B, N)`) — see `GLiNERBaseOutput` construction at
base.py:689-699.

### 3d. Bi-encoder, **token** mode — `BiEncoderTokenModel`

File: `gliner/modeling/base.py:1073` (`class BiEncoderTokenModel(BaseBiEncoderModel,
UniEncoderTokenModel)`, `forward` at base.py:1093). Reuses `UniEncoderTokenModel`'s `Scorer` and
scoring logic via MRO — same **`(B, W, C, 3)`** shape as 3c, again with the separate label
encoder for `prompts_embedding`.

### 3e. Decoder variants (`UniEncoderSpanDecoderModel`, `UniEncoderTokenDecoderModel`)

`gliner/modeling/base.py:1199` (`forward` at 1515) and `:1706` (`forward` at 1865). These wrap the
span/token model above and additionally run a generative label decoder
(`gliner/modeling/decoder.py`) that produces label *text* (not scores) for each detected span;
the underlying span-score tensor going into the generative stage is still the `(B, L, K, C)` /
`(B, W, C, 3)` tensor from 3a/3c. Output is `GLiNERDecoderOutput` (outputs.py:44-72), which adds
`decoder_loss`, `decoder_embedding`, `decoder_span_idx` fields but keeps `logits` semantics
identical to the base span/token model.

### 3f. Relex variants (`UniEncoderSpanRelexModel`, `UniEncoderTokenRelexModel`)

`gliner/modeling/base.py:2086` (`forward` at 2256) and `:2621`. Adds relation-extraction outputs
on top of the standard NER `logits` tensor: `GLiNERRelexOutput` (outputs.py:76-108) adds
`rel_idx` `(B, num_relations, 2)`, `rel_logits` `(B, num_relations, num_relation_types)`,
`rel_mask`, `entity_spans`. The entity-level `logits` field is still the same span/token tensor
as 3a/3c depending on `span_mode`.

### Summary table

| Path | Class | `logits` shape | Notes |
|---|---|---|---|
| Uni-encoder, span | `UniEncoderSpanModel` (base.py:383) | `(B, L, K, C)` | `einsum` at base.py:474 |
| Bi-encoder, span | `BiEncoderSpanModel` (base.py:889) | `(B, L, K, C)` | `einsum` at base.py:991 |
| Uni-encoder, token | `UniEncoderTokenModel` (base.py:560) | `(B, W, C, 3)` | `Scorer` at base.py:672; optional extra `span_logits` `(B, N, C)` at base.py:678 |
| Bi-encoder, token | `BiEncoderTokenModel` (base.py:1073) | `(B, W, C, 3)` | same `Scorer` path via MRO |
| Uni-encoder span/token + decoder | `UniEncoderSpanDecoderModel`/`UniEncoderTokenDecoderModel` | same as above | adds generative decoder outputs, doesn't change span-score shape |
| Uni-encoder span/token + relex | `UniEncoderSpanRelexModel`/`UniEncoderTokenRelexModel` | same as above | adds `rel_logits` `(B, num_rel, num_rel_types)` |

**No poly-encoder path exists** (see §2).

---

## 4. `predict_entities` / `batch_predict_entities` — sigmoid, threshold, decoding

Both live on `BaseEncoderGLiNER` in `gliner/model.py`:

- `predict_entities(text, labels, flat_ner=True, threshold=0.5, multi_label=False,
  return_class_probs=False, **kwargs)` — model.py:2340-2372. Thin wrapper: calls
  `self.inference([text], labels, ...)[0]`.
- `batch_predict_entities(texts, labels, flat_ner=True, threshold=0.5, multi_label=False,
  **kwargs)` — model.py:2374-2414. **Deprecated** (`FutureWarning` at model.py:2401-2406,
  "will be removed in a future release"); forwards to `self.inference(...)`.
- The real entrypoint is `inference(texts, labels, flat_ner=True, threshold=0.5,
  multi_label=False, batch_size=8, ...)` — model.py:2259-2338 (decorated `@torch.no_grad()`
  at model.py:2259).

`inference` calls, in order: `prepare_batch` → `create_collator`/`collate_batch` (via
`DataLoader`) → `self._process_batches(...)` (model.py:2318-2327) → `map_entities_to_text`
(model.py:2329-2336).

`_process_batches` (model.py:1982-2023) is the loop that, per batch, calls:
1. `self.run_batch(batch, threshold=threshold, ...)` (model.py:1998-2004) → raw model forward
   pass, `@torch.inference_mode()` (model.py:2134), returns the `GLiNERBaseOutput` (or subclass)
   with **un-sigmoided, unthresholded** logits (model.py:2165: `model_output =
   self.model(**model_inputs, threshold=threshold)`).
2. `self.decode_batch(model_output, batch, threshold=threshold, flat_ner=flat_ner,
   multi_label=multi_label, ...)` (model.py:2012-2020) → this is where sigmoid + threshold +
   greedy decoding actually happen, delegated to `self.decoder.decode(...)`
   (model.py:2196-2208), where `self.decoder` is one of `SpanDecoder` / `TokenDecoder` /
   `SpanRelexDecoder` / `TokenRelexDecoder` / `SpanGenerativeDecoder` / `TokenGenerativeDecoder`
   from `gliner/decoding/` (chosen via `decoder_class` set on each concrete `*GLiNER` class,
   see model.py imports at 46-53).

### Sigmoid + threshold, concretely (span path)

`gliner/decoding/decoder.py`, class `BaseSpanDecoder`:
- `decode(...)` (decoder.py:475-524): `probs = torch.sigmoid(model_output)` at **decoder.py:509**
  — this is the sigmoid application point for the `(B, L, K, C)` span-score tensor.
- Threshold comparison happens in `_decode_batch` (decoder.py:332-473) via
  `torch.where(probs > threshold_tensor)` at **decoder.py:413** (batched path) or via
  `_find_candidate_spans`, `torch.where(probs > threshold)` at **decoder.py:163** (single-item
  path, `BaseSpanDecoder._find_candidate_spans`, decoder.py:140-163).

### Sigmoid + threshold (token path)

`gliner/decoding/decoder.py`, class `TokenDecoder` (decoder.py:1196 onward):
- Token-level (BIO start/end/inside) decode: `_get_indices_above_threshold` (decoder.py:1204-1216)
  does `scores = torch.sigmoid(scores)` (decoder.py:1215) then `torch.where(scores > threshold)`
  (decoder.py:1216). Final per-span score is the **minimum** of the start/end/inside scores for
  that span (decoder.py:1268: `spn_score = min(*ins, start_score, end_score)`) — i.e. token-mode
  span confidence is a min-pooling over 3 sigmoid probabilities, not a single logit.
- Span-level decode (when `represent_spans=True`, using `output.span_logits`):
  `_decode_from_spans` (decoder.py:1272-1355) does `span_probs = torch.sigmoid(span_logits)` at
  **decoder.py:1316**, then a plain Python threshold comparison `if prob <= threshold_i: continue`
  (decoder.py:1345).

### `flat_ner` — flat vs. nested/overlapping resolution

All decoders share `BaseDecoder.greedy_search(spans, flat_ner=True, multi_label=False)`
(`gliner/decoding/decoder.py:92-137`): sorts candidate `Span` objects by `-score` (descending,
decoder.py:121), then greedily keeps a span only if it doesn't overlap any already-kept span,
using either `has_overlapping` (flat_ner=True — **no overlaps or nesting allowed**) or
`has_overlapping_nested` (flat_ner=False — **nesting allowed, only true partial-overlaps
rejected**), both defined in `gliner/decoding/utils.py:6-19`:

```python
def has_overlapping(idx1, idx2, multi_label=False):        # utils.py:6
    if idx1[:2] == idx2[:2]:
        return not multi_label
    return not (idx1[0] > idx2[1] or idx2[0] > idx1[1])

def has_overlapping_nested(idx1, idx2, multi_label=False): # utils.py:14
    if idx1[:2] == idx2[:2]:
        return not multi_label
    return not ((idx1[0] > idx2[1] or idx2[0] > idx1[1]) or is_nested(idx1, idx2))
```

`is_nested` (utils.py:1-3) checks strict containment either direction.

Default for `predict_entities`/`inference`: `flat_ner=True` (model.py:2344, :2264). Default for
`evaluate`: `flat_ner=False` (model.py:2420) — i.e. eval by default allows nested spans, live
inference defaults to flat.

---

## 5. Earliest interception point for RAW per-span scores

The pipeline stage boundary that matters for a conformal wrapper:

```
run_batch()          →  model_output = self.model(**model_inputs, threshold=threshold)
                         [model.py:2165]  --- RAW, PRE-SIGMOID LOGITS, PRE-THRESHOLD, PRE-DECODE ---
                         GLiNERBaseOutput.logits: (B,L,K,C) span-mode / (B,W,C,3) token-mode
                         (+ .span_logits/.span_idx/.span_mask when represent_spans=True)
                            │
                            ▼
decode_batch()        →  self.decoder.decode(...)  [model.py:2196]
                            │
                            ├─ sigmoid: decoder.py:509 (span) / decoder.py:1215,1316 (token)
                            ├─ threshold filter (torch.where / prob <= threshold): decoder.py:413/163/1216/1345
                            └─ greedy_search overlap resolution: decoder.py:92-137
                            │
                            ▼
                         List[List[Span]]  --- COLLAPSED: only surviving, non-overlapping spans ---
```

**The cleanest interception point is immediately after `run_batch()` returns, i.e. the
`GLiNERBaseOutput`/`GLiNERDecoderOutput`/`GLiNERRelexOutput` object itself (or equivalently,
before `decode_batch()`/`self.decoder.decode(...)` is invoked).** At that point:

- For span-mode models: `model_output.logits` is the full dense `(B, L, K, C)` (or `(B, W, C, 3)`
  for token-mode) raw score tensor for **every** candidate span/type pair, not just those that
  survive thresholding — this is exactly the object a conformal calibration/prediction-set
  procedure needs (full score distribution over the label set per span, pre-decision).
- For token-mode models with `represent_spans=True`, `model_output.span_logits` /
  `.span_idx` / `.span_mask` give the analogous dense per-span-per-class raw scores.
- `model.py`'s own `decode_batch` (model.py:2168-2209) already threads exactly this object
  (`model_output[0]` i.e. `model_output.logits`, plus `.span_idx`/`.span_mask`/`.span_logits`)
  into `self.decoder.decode(...)` — so a `ConformalGLiNER` wrapper can call
  `self.run_batch(batch, threshold=..., ...)` directly, work with `model_output.logits` (applying
  its own sigmoid/softmax and conformal nonconformity score), and only call (a modified) decode
  logic afterward, or bypass `self.decoder.decode` entirely and write its own conformal-aware
  candidate-set construction reusing `greedy_search`/`has_overlapping[_nested]` from
  `gliner/decoding/utils.py` for the flat-NER collapsing step.
- No existing code currently exposes `run_batch`'s output directly to callers of
  `predict_entities`/`inference` — `_process_batches` (model.py:1982-2023) always chains
  `run_batch` immediately into `decode_batch` and only returns the final decoded `Span` list. So
  raw scores are *technically* reachable today (both methods are public, undecorated with `_`)
  but there is no supported one-call API that returns them — a conformal wrapper calling
  `run_batch` + a custom decode path is the correct, minimally-invasive approach; **no changes to
  existing model code are required** to get raw scores (confirms the "fully additive, no core
  changes" premise in the mission brief).

---

## 6. Where evaluation (F1/precision/recall) lives

- `gliner/evaluation/evaluator.py`:
  - `BaseEvaluator` (evaluator.py:9-129), abstract, with `compute_prf(y_true, y_pred,
    average="micro")` static method (evaluator.py:33-91) — computes precision/recall/F1 via
    `extract_tp_actual_correct`/`_prf_divide` (imported from `gliner/evaluation/utils.py`).
  - `BaseNEREvaluator(BaseEvaluator)` (evaluator.py:132-194) — entity-level exact-match
    evaluation: an entity is correct only if `(label, (start, end))` matches exactly
    (`get_predictions`, evaluator.py:156-173, reads `ent.entity_type`/`ent.start`/`ent.end` off
    `Span` objects or raw tuples).
  - `BaseRelexEvaluator(BaseEvaluator)` (evaluator.py:197-282) — relation-level exact-match
    evaluation (head span + tail span + relation label).
- `gliner/evaluation/evaluate_ner.py` (330 lines) — standalone dataset-loading + scripted
  evaluation harness (`open_content`, `process`, etc.) for CoNLL-style benchmark directories, used
  by `benchmarks/` scripts, not part of the core model API.
- Model-level entrypoint: `BaseEncoderGLiNER.evaluate(test_data, flat_ner=False, multi_label=False,
  threshold=0.5, batch_size=12, entity_types=None)` — `gliner/model.py:2416-2460`. Runs
  `_process_batches` to get predictions, then `evaluator = BaseNEREvaluator(all_trues, all_preds);
  out, f1 = evaluator.evaluate()` (model.py:2457-2458). Note: `evaluate()`'s default `flat_ner`
  is `False` (nested allowed) whereas `predict_entities`/`inference` default to `True`.

For a conformal wrapper, coverage/efficiency evaluation will likely need a new evaluator (not
reuse `BaseNEREvaluator` as-is, since it expects a single decoded entity list per example, not a
prediction *set* with a size/coverage notion) — but `compute_prf`'s TP/FP/FN machinery in
`gliner/evaluation/utils.py` (`extract_tp_actual_correct`, `flatten_for_eval`) may still be
reusable for reporting standard P/R/F1 alongside conformal coverage metrics.

---

## 7. Config system

File: `gliner/config.py`. `BaseGLiNERConfig(PretrainedConfig)` (config.py:7-116) is the root;
`is_composition = True`, registered into `transformers.models.auto.CONFIG_MAPPING` at the bottom
of the file (config.py:371-386) under keys like `"gliner_uni_encoder_span"`,
`"gliner_bi_encoder_token"`, etc. (these are the `model_type` strings each subclass sets, e.g.
config.py:134, :143, :304, :313).

Fields most relevant to a conformal wrapper (all on `BaseGLiNERConfig.__init__`,
config.py:13-116):

- `max_width: int = 12` (config.py:17) — max span width `K` in the span-mode `(B, L, K, C)` score
  tensor (§3a/3b). Directly determines how many candidate spans exist per start position.
- `max_types: int = 25` (config.py:27) — max number of entity types considered together in one
  forward pass (i.e. an upper bound on `C`, the per-call type-prompt budget); also
  `max_neg_type_ratio: int = 1` (config.py:26) controls negative-type sampling ratio during
  training (not inference-relevant).
- `max_len: int = 384` (config.py:28) — max input sequence length (subword tokens).
- `id_to_classes: Optional[dict] = None` (config.py:43) — the runtime class-id → label-name map;
  populated per-inference-call by the data collator, also settable persistently via
  `compress_prompt_embeddings`/`_compute_prompt_embeddings` (model.py:2656-2657) for
  precomputed-prompt mode.
- `span_mode: str = "markerV0"` (config.py:22) — selects span representation scheme; forced to
  `"token_level"` by `UniEncoderTokenConfig`/`BiEncoderTokenConfig`/relex-token variants
  (config.py:142, :312, :272) to route into token-mode models (§3c/3d).
  `GLiNERConfig._resolve_model_type()` (config.py:350-367) uses `span_mode == "token-level"` (note
  hyphen, not underscore — worth flagging as a possible existing inconsistency, though not this
  agent's job to fix) plus presence of `labels_decoder`/`labels_encoder`/`relations_layer` to
  auto-select the concrete `model_type`.
- `precomputed_prompts_mode: Optional[bool] = None` (config.py:42) — when True, skips
  label-prompt-prepending/encoding per call and looks up cached per-label embeddings instead;
  relevant if a conformal wrapper wants deterministic/cacheable label representations across
  calibration and test-time inference.
- Per-architecture extensions: `UniEncoderSpanDecoderConfig` adds `decoder_mode`
  ("prompt"/"span"), `labels_decoder`, `blank_entity_prob` (config.py:149-186);
  `UniEncoderRelexConfig` adds `relations_layer`, `rel_token_index`, `rel_id_to_classes`, and data
  augmentation knobs (config.py:196-254); `BiEncoderConfig` adds `labels_encoder`/
  `labels_encoder_config` (config.py:275-294).

`GLiNERConfig` (config.py:316-367) is the "legacy"/convenience config that auto-resolves
`model_type` from which of `labels_encoder`/`labels_decoder`/`relations_layer`/`span_mode` are
set — this is what `gliner/__init__.py` exports and what most `from_pretrained` calls implicitly
construct via `_load_config`.

---

## 8. ONNX export paths (brief)

`gliner/onnx/model.py` defines an ORT-backed mirror of the model hierarchy: `BaseORTModel(ABC)`
(onnx/model.py:20), with concrete `UniEncoderSpanORTModel`, `BiEncoderSpanORTModel`,
`UniEncoderTokenORTModel`, `BiEncoderTokenORTModel`, `UniEncoderSpanRelexORTModel`,
`UniEncoderTokenRelexORTModel` (onnx/model.py:114, 161, 223, 264, 321, 374). `BaseGLiNER` treats
an ONNX-backed model as functionally interchangeable with the PyTorch one — `self.onnx_model =
isinstance(self.model, BaseORTModel)` (model.py:154-157), and `run_batch`/`device` branch on this
flag (model.py:2155, :216-222). `from_pretrained(..., load_onnx_model=True, onnx_model_file=
"model.onnx")` loads the ORT session instead of PyTorch weights (model.py:1057-1058, referenced
again around :1191). Not investigated further — flagged as out of scope per the task brief, but
worth knowing that a conformal wrapper's raw-score interception point (`run_batch`'s return value,
§5) is architecturally the same for both backends since `decode_batch` doesn't care whether
`model_output` came from PyTorch or ONNX (model.py:2192-2194 explicitly handles the numpy-vs-tensor
case: `if not isinstance(model_logits, torch.Tensor): model_logits = torch.from_numpy(model_logits)`).

---

## 9. Test setup and conventions

Directory: `tests/` (no `conftest.py` exists anywhere in the repo — confirmed by directory
listing). Files present: `test_data_processing.py`, `test_decoder.py`, `test_features_selection.py`,
`test_infer_packing.py`, `test_local_files_only.py`, `test_modeling.py`, `test_models.py`,
`test_quantize_and_dtype.py`, `test_serve.py`, `test_tokenizer_stanza.py`, `utils_infer.py`
(shared helper module, not a test file itself — imported via absolute `tests.utils_infer` in
`test_infer_packing.py`).

Pytest config: `pyproject.toml:77-82`:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```
No custom markers, no `--no-network`/`vcr`-style gating configured. Dev dependency group
(`pyproject.toml:71-75`) is just `pytest`, `pytest-asyncio`, `ruff` — no `pytest-mock`,
`responses`, or HF-mocking libraries.

**Small pretrained model download pattern**: `tests/test_models.py:23-34`
(`test_span_model`) calls `GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")` directly
and unconditionally at test time — no fixture, no caching layer beyond whatever the default HF
Hub cache (`huggingface_hub` default `~/.cache/huggingface`) provides via `snapshot_download`
under the hood. This is the **only** test in the suite that hits the network / needs a real
pretrained checkpoint; every other test in `test_models.py` uses a hand-built "minimal" model via
`_minimal_encoder_model()` (test_models.py:17-20), which does `cls.__new__(cls)` and manually
stubs `data_processor` — bypassing `from_pretrained` and any weight loading entirely, for testing
pure Python logic (`prepare_batch` etc.) without touching the network or a real encoder.

Fixture conventions elsewhere (`pytest.fixture`, plain function-scoped, no custom scope
declarations found):
- `tests/test_decoder.py` — heavy use of `@pytest.fixture` for hand-built config objects and
  synthetic tensors (`basic_config`, `basic_inputs`, `relex_config`, `token_config`, etc.) to unit
  test `gliner/decoding/decoder.py` classes directly without any real model — this is the closest
  existing precedent for how conformal-prediction unit tests (`test_conformal*.py`) should be
  structured: synthetic logits tensors + hand-built minimal configs, no network/model download.
- `tests/test_local_files_only.py` — `@pytest.fixture` for `config`/`mock_tokenizer`, uses
  `unittest.mock.patch` on `gliner.model.AutoTokenizer.from_pretrained` to avoid real downloads.
- `tests/test_modeling.py` — `@pytest.fixture` (`basic_setup`, `prompt_setup`) building small
  synthetic tensors to test `gliner/modeling/` layers directly (e.g. `extract_prompt_features`)
  without a full model.

Naming convention: `test_<module_area>.py` mirroring the `gliner/` submodule under test
(`test_decoder.py` ↔ `gliner/decoding/decoder.py`, `test_modeling.py` ↔ `gliner/modeling/`,
`test_data_processing.py` ↔ `gliner/data_processing/`). A future `tests/test_conformal.py` (or
`test_conformal_calibrators.py` + `test_conformal_gliner.py` if split by unit) fits this
convention directly. Given `test_decoder.py`'s pattern (synthetic tensors, no real model needed
for the calibrator math), the calibration-logic unit tests should not need network access at all;
only an end-to-end integration test analogous to `test_models.py::test_span_model` would need the
`gliner-community/gliner_small-v2.5` real-download pattern.

---

## 10. Existing "conformal"/"calibrat"/"confidence"/"uncertainty" references

Grepped case-insensitively across the whole repo (`*.py`, `*.md`, `*.rst`), excluding this
branch's own scratch docs (`/CLAUDE.md`, `ROADMAP.md`, `docs/archive/mission_brief.md`, which are
this project's own planning artifacts, not pre-existing upstream content):

- **"conformal"**: zero hits anywhere in the codebase outside this branch's own planning docs.
  Confirmed nothing pre-exists to build on or conflict with.
- **"uncertainty"**: zero hits anywhere.
- **"calibrat"**: hits exist, but every single one is about **GPU-memory calibration for the Ray
  Serve layer** — completely unrelated to statistical/conformal calibration:
  - `gliner/serve/memory.py` (module docstring line 1: "Memory estimation for GLiNER via
    precomputed calibration table"; `calibrate()` method at memory.py:82).
  - `gliner/serve/server.py:282-292` (`_calibrate_memory`, "Calibrating memory table...").
  - `gliner/serve/config.py:52-53` (`calibration_min_seq_len`, `calibration_probe_batch_size`).
  - `README.md:148` ("memory-aware batch sizing that prevents CUDA OOM by calibrating against
    your GPU").
  - `docs/usage.md:1250-1296` uses `calibration_texts` as a variable name for the corpus passed to
    `compress_prompt_embeddings` (§2462 in model.py) — i.e. "calibration" there means
    "texts used to average/compute prompt embeddings", not statistical calibration either.
- **"confidence"**: many hits, but they are uniformly the generic phrase "confidence threshold" /
  "confidence score" in docstrings for the existing `threshold: float = 0.5` parameter (e.g.
  `predict_entities` docstring at model.py:2356, `Span.score` docstring at decoder.py:36,
  `TokenDecoder._get_indices_above_threshold` docstring at decoder.py:1210) — not a calibrated
  confidence in any statistical sense, just the raw post-sigmoid probability compared against the
  fixed 0.5 default.

**Conclusion: there is no prior art, partial implementation, or naming collision to worry about.**
The `gliner.conformal` (or similar) namespace, `ConformalGLiNER` class name, and any
`calibrate()`/`calibration_set` API surface a Phase-2 implementation introduces will not shadow or
conflict with anything that already exists in this checkout.
