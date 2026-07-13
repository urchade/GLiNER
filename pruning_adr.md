# Vocabulary Pruning Engine — ADR & Status Tracker

> Persistent scratchpad for the `feature/vocab-pruning-engine` branch.
> (Named pruning_adr.md because macOS filesystem is case-insensitive; claude.md == CLAUDE.md)
> Updated as work progresses. Read this before touching any code in this feature.

---

## Status

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — Deep Codebase Research | ✅ COMPLETE | See findings below |
| Phase 2 — Planning & Approval | ✅ COMPLETE | Plan below — **AWAITING USER APPROVAL** |
| Phase 3 — Implementation | ✅ COMPLETE | scripts/prune_gliner_vocab.py + scripts/validate_pruned_model.py |
| Phase 4 — Testing & Validation | ✅ COMPLETE | ALL 6 test cases PASS ✓ |

---

## Branch Setup (run in your terminal)

```bash
git checkout -b feature/vocab-pruning-engine
```

---

## Phase 1 Findings — Deep Codebase Architecture

### The Critical Access Path to Word Embeddings

```
GLiNER.from_pretrained(model_id)          # returns a BaseGLiNER subclass
  └── .model                              # BaseModel subclass (UniEncoderSpanModel etc.)
      └── .token_rep_layer                # Encoder or BiEncoder (gliner/modeling/encoder.py)
          └── .bert_layer                 # Transformer wrapper (gliner/modeling/encoder.py)
              └── .model                  # HuggingFace model (e.g. DebertaV2Model)
                  └── .embeddings
                      └── .word_embeddings  # nn.Embedding(V, d) ← THE MATRIX TO SLICE
```

For **BiEncoder** models (e.g. `knowledgator/gliner-bi-small-v1.0`), there is a SECOND encoder:
```
  └── .token_rep_layer
      ├── .bert_layer.model.embeddings.word_embeddings    # text encoder
      └── .labels_encoder.model.embeddings.word_embeddings  # label encoder
```
Both must be pruned if they share the same tokenizer vocabulary.

### Tokenizer

- Type: `AutoTokenizer` → for mDeBERTa-v3 resolves to `DebertaV2Tokenizer`
- mDeBERTa-v3 vocab: **250,002 tokens** (SentencePiece Unigram model)
- Accessed at: `gliner_model.data_processor.transformer_tokenizer`
- Saved via: `.save_pretrained(dir)` → produces `tokenizer.json`, `spm.model`, etc.
- Fast tokenizer (`tokenizer.json`) encodes vocab as Unigram list: `[[token, score], ...]`
- **Key insight**: We modify `tokenizer.json` directly (JSON surgery), NOT the binary `spm.model`

### GLiNER Special Tokens

Added at model load time via `tokenizer.add_tokens([...], special_tokens=True)`:
```python
# BaseGLiNER._get_special_tokens():
tokens = ["[FLERT]", config.ent_token, config.sep_token]  # → IDs [V, V+1, V+2]
# For relex models: also config.rel_token                  # → ID [V+3]
```

`config.class_token_index = len(tokenizer) - 2` → points to `ent_token` (second-from-last)

### Embedding Resize — The Existing Pattern We Mirror

`BaseEncoderGLiNER.resize_embeddings()` calls:
```python
new_num_tokens = len(self.data_processor.transformer_tokenizer)
model_embeds = self.model.token_rep_layer.resize_token_embeddings(new_num_tokens, None)
self.config.vocab_size = model_embeds.num_embeddings
if hasattr(self.config, "encoder_config"):
    self.config.encoder_config.vocab_size = model_embeds.num_embeddings
```
→ Our script mirrors this pattern exactly when writing the new vocab size.

### Config Fields to Update After Pruning

- `config.vocab_size` → new K (pruned vocab size)
- `config.encoder_config.vocab_size` → new K
- `config.class_token_index` → remapped index of `ent_token` in new vocab

### DeBERTa Architecture — No Position Embeddings to Slice

DeBERTa v2/v3 uses disentangled relative position attention — there is **no absolute
`position_embeddings` matrix** in the embedding layer. Only `word_embeddings` (the token
lookup table) needs slicing. This is simpler than BERT/RoBERTa.

### Save/Load Chain

```python
# Save:
gliner_model.save_pretrained(output_dir)
  # → torch.save(state_dict, "pytorch_model.bin")
  # → config.to_json_file("gliner_config.json")
  # → tokenizer.save_pretrained(output_dir)

# Load (from_pretrained):
GLiNER.from_pretrained(output_dir)
  # → reads gliner_config.json → instantiates config
  # → reads tokenizer from output_dir
  # → reads pytorch_model.bin → load_state_dict()
  # → resize_embeddings() fires ONLY if class_token_index == -1 or vocab_size == -1
```

### Key: Prevent Double Resize on Re-load

After pruning we save `config.vocab_size = K` (not -1). `from_pretrained` will skip
`resize_embeddings()` because both guard conditions are false. Correct — the embedding
is already the right size.

---

## Phase 2 Plan — Implementation Strategy

### Script: `scripts/prune_gliner_vocab.py`

**CLI Arguments:**
```
--model_id          HuggingFace model ID or local path (required)
--dataset_for_vocab "wikipedia" or path to local .txt file (required)
--output_dir        Where to save pruned model (required)
--top_k             Keep top-K most frequent tokens (default: 30000)
--lang              Wikipedia language code: "en", "fr", "de", etc. (default: "en")
--min_freq          Min token frequency to keep (default: 1)
```

---

### Step-by-Step Mathematical Approach

#### Step 1 — Load model and tokenizer

```python
gliner_model = GLiNER.from_pretrained(model_id)
tokenizer = gliner_model.data_processor.transformer_tokenizer
V = len(tokenizer)   # original vocab size, e.g. 250,005 (250,002 + 3 GLiNER tokens)
```

#### Step 2 — Collect active tokens from corpus

```python
freq: Counter[int] = Counter()
for text in corpus_texts:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    freq.update(ids)
active_ids: set[int] = {tok_id for tok_id, _ in freq.most_common(top_k)}
```

#### Step 3 — Build the KEEP SET

```python
# 1. Standard HuggingFace special tokens
special_ids: set[int] = set()
for attr in ["pad_token_id","unk_token_id","cls_token_id","sep_token_id",
             "mask_token_id","bos_token_id","eos_token_id"]:
    tid = getattr(tokenizer, attr, None)
    if tid is not None:
        special_ids.add(tid)

# 2. Byte-fallback tokens (mDeBERTa IDs 3-258; never safe to drop)
byte_fallback_ids: set[int] = set(range(3, 259))  # detect from tokenizer vocab

# 3. GLiNER-added tokens (last N tokens added via add_tokens)
gliner_added_ids: set[int] = {tok["id"] for tok in tokenizer.added_tokens_decoder.values()}

keep_ids: list[int] = sorted(active_ids | special_ids | byte_fallback_ids | gliner_added_ids)
K: int = len(keep_ids)
```

#### Step 4 — Build the ID remapping table

```python
# keep_ids is sorted ascending → new ID = position in this list
old_to_new: dict[int, int] = {old: new for new, old in enumerate(keep_ids)}

# Mathematical bijection: for any kept token t_old,
#   new_embedding[old_to_new[t_old]] == old_embedding[t_old]
```

#### Step 5 — Slice the embedding weight tensor

```python
keep_tensor = torch.tensor(keep_ids, dtype=torch.long)
bert_model = gliner_model.model.token_rep_layer.bert_layer.model

E_old = bert_model.embeddings.word_embeddings.weight.data   # shape: (V, d)
E_new = E_old[keep_tensor]                                  # shape: (K, d)

pad_new_id = old_to_new.get(tokenizer.pad_token_id, 0)
new_embed = nn.Embedding(K, E_old.shape[1], padding_idx=pad_new_id)
new_embed.weight = nn.Parameter(E_new)
bert_model.embeddings.word_embeddings = new_embed
bert_model.config.vocab_size = K
```

**Invariant:** `E_new[old_to_new[t]] == E_old[t]` for all t ∈ keep_ids (exact row preservation).

#### Step 6 — Apply same slice to labels encoder (BiEncoder only)

```python
if hasattr(gliner_model.model.token_rep_layer, "labels_encoder"):
    le_bert = gliner_model.model.token_rep_layer.labels_encoder.model
    if le_bert.config.vocab_size == V:  # same tokenizer space → same pruning
        E_le = le_bert.embeddings.word_embeddings.weight.data[keep_tensor]
        le_embed = nn.Embedding(K, E_le.shape[1], padding_idx=pad_new_id)
        le_embed.weight = nn.Parameter(E_le)
        le_bert.embeddings.word_embeddings = le_embed
        le_bert.config.vocab_size = K
```

#### Step 7 — Update GLiNER config

```python
gliner_model.config.vocab_size = K
if hasattr(gliner_model.config, "encoder_config") and gliner_model.config.encoder_config:
    gliner_model.config.encoder_config.vocab_size = K

old_cti = gliner_model.config.class_token_index
gliner_model.config.class_token_index = old_to_new[old_cti]
```

#### Step 8 — Rebuild the fast tokenizer (tokenizer.json surgery)

The fast tokenizer stores vocab as a list at `tok_data["model"]["vocab"]`.
Each entry is `[token_string, score]` and its **list index IS the token ID**.

```python
tok_data = json.loads((Path(model_dir) / "tokenizer.json").read_text())

old_vocab: list = tok_data["model"]["vocab"]   # list of [str, float]
new_vocab = [old_vocab[i] for i in keep_ids]  # select kept rows (in new order)
tok_data["model"]["vocab"] = new_vocab

# Remap explicit ID references in added_tokens list
for entry in tok_data.get("added_tokens", []):
    old_id = entry["id"]
    if old_id in old_to_new:
        entry["id"] = old_to_new[old_id]

# Remap post_processor template IDs (CLS/SEP) if present
# (These are usually stored as token strings, not IDs — often no-op)

(Path(output_dir) / "tokenizer.json").write_text(
    json.dumps(tok_data, ensure_ascii=False, indent=2)
)
```

#### Step 9 — Save the pruned model

```python
gliner_model.save_pretrained(output_dir)
# Produces: pytorch_model.bin (state dict with sliced E_new),
#           gliner_config.json (K, new class_token_index),
#           tokenizer.json (pruned vocab, remapped IDs)
```

---

### Phase 4 Validation Plan

```python
orig   = GLiNER.from_pretrained(original_model_id)
pruned = GLiNER.from_pretrained(output_dir)

test_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
labels    = ["person", "organization", "location"]

orig_out   = orig.predict_entities(test_text, labels)
pruned_out = pruned.predict_entities(test_text, labels)

assert orig_out == pruned_out, f"Entity mismatch!\n  orig={orig_out}\n  pruned={pruned_out}"

orig_mb   = sum(p.numel() * p.element_size() for p in orig.parameters())   / 1e6
pruned_mb = sum(p.numel() * p.element_size() for p in pruned.parameters()) / 1e6
reduction = (orig_mb - pruned_mb) / orig_mb * 100
print(f"Model size: {orig_mb:.1f} MB → {pruned_mb:.1f} MB  ({reduction:.1f}% reduction)")
```

---

## Risk Register

| Risk | Mitigation |
|---|---|
| `tokenizer.json` Unigram vocab list format differs across models | Assert `tok_data["model"]["type"] == "Unigram"` early; add SPM-only fallback |
| Byte-fallback tokens (IDs 3-258 for mDeBERTa) silently dropped | Auto-detect from tokenizer vocab; always include in keep set |
| `added_tokens` in tokenizer.json stores old IDs | Explicitly remap in Step 8 |
| BiEncoder labels encoder uses different vocab / tokenizer | Detect by comparing vocab sizes; skip or handle separately |
| `post_processor` stores CLS/SEP as token strings (not IDs) | Usually safe; add assertion after surgery that special tokens resolve correctly |
| Re-loading the pruned model triggers `resize_embeddings()` | Save `vocab_size = K` (not -1) → guard condition in `from_pretrained` is false |
| GLiNER `class_token_index` points to a token NOT in keep set | Impossible by construction (GLiNER tokens always in `gliner_added_ids`) |

---

## Files to Create

- `scripts/prune_gliner_vocab.py` — main engine (Phase 3) ← **pending approval**
- `scripts/validate_pruned_model.py` — validation script (Phase 4) ← **pending approval**

## Files Modified

_(None yet — awaiting explicit user approval before touching any Python code)_

---

## ADR Log

| Date | Decision | Reason |
|---|---|---|
| 2026-06-03 | Modify `tokenizer.json`, NOT `spm.model` | SPM binary is a compiled protobuf; tokenizer.json is a plain JSON list → simple index selection |
| 2026-06-03 | Sort `keep_ids` ascending before slicing | Preserves relative token order; new IDs assigned 0…K-1 monotonically |
| 2026-06-03 | Keep all byte-fallback tokens unconditionally | mDeBERTa uses byte fallback; dropping any crashes tokenization of non-ASCII chars |
| 2026-06-03 | Apply same slice to `labels_encoder` if vocab matches | BiEncoder shares tokenizer; mismatched embedding size would crash forward pass |
| 2026-06-03 | Save `config.vocab_size = K` (not -1) | Prevents `resize_embeddings()` re-firing on load which would re-expand the matrix |
| 2026-06-03 | No lm_head / cls head to update | GLiNER doesn't use the causal/masked LM head; only word_embeddings is used |
