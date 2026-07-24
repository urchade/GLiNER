# Input limits and truncation

GLiNER does not automatically split a long document into windows. For ordinary
stateless inference, each input that exceeds the checkpoint's `config.max_len`
is reduced to a prefix. In GLiNER 0.2.27 this produces a `UserWarning`, but the
prediction call still succeeds and its return value does not say that part of
the input was skipped.

This behavior applies to span, token, bi-encoder, decoder, and relation
extraction models. Cached StreamingSpan sessions are the exception described
under **Architecture-specific limits** below.

## What `config.max_len` counts

`config.max_len` is the maximum number of **text tokens produced by GLiNER's
word splitter**. It is not a character count, a whitespace-word count, a
transformer subword count, or the total prompt-plus-text sequence length.

For raw-text inference, the processing order is:

```text
raw text
  -> WordsSplitter(config.words_splitter_type)
  -> N text tokens
  -> if N > config.max_len: warn and keep tokens[:config.max_len]
  -> add the entity/relation-type prompt
  -> transformer subword tokenization
  -> model
```

The default `whitespace` splitter also separates punctuation. For example,
`"Acme, Inc."` becomes four splitter tokens (`Acme`, `,`, `Inc`, `.`), even
though a simple `text.split()` returns two strings. Other
`words_splitter_type` values segment text differently. Always inspect the
loaded checkpoint rather than assuming the configuration default:

```python
print(model.config.max_len)
print(model.config.words_splitter_type)
```

The base configuration defaults to `max_len=384`, but checkpoints can save a
different value. For example,
`EmergentMethods/gliner_medium_news-v2.1` saves `max_len=296`.

## Do labels reduce the text budget?

At the `config.max_len` stage, **no**. GLiNER splits and truncates the text
first, then prepends or separately encodes the entity-type prompt. Ten labels
and one hundred labels therefore receive the same `config.max_len` allowance
for text.

There is a second, independent limit to consider. A transformer sees
subtokens, and uni-encoder architectures put the label prompt and retained text
in one transformer sequence. Prompt subtokens therefore consume part of any
finite tokenizer or backbone context capacity. A finite tokenizer limit can
truncate the text tail; a backbone limit can instead reject an oversized
combined sequence. Either way, adding labels reduces the remaining combined
headroom even though it does not change `config.max_len`.

GLiNER calls the transformer tokenizer with `truncation=True` but without an
explicit tokenizer `max_length`, so the tokenizer's registered limit and
truncation side determine tokenizer-level behavior. Some tokenizers do not
register a finite limit; Transformers may then warn that no maximum was
provided and perform no tokenizer-level truncation. That does not prove that
the backbone supports an unlimited sequence.

## Behavior when text exceeds `max_len`

For each input independently, when `num_tokens > model.config.max_len`, GLiNER
0.2.27:

1. emits a `UserWarning` such as
   `Sentence of length 987 has been truncated to 296`;
2. retains only the first `max_len` splitter tokens;
3. runs inference normally on that prefix; and
4. returns ordinary-looking predictions with no truncation metadata.

Text after the retained prefix is never presented to the model. Entities there
cannot be returned, and entities crossing the boundary are incomplete. The
same prefix truncation occurs during training when a pre-tokenized example is
too long.

Python warning filters control whether the warning is visible. With the default
filter, repeated warnings from the same location may be shown only once per
process. Warnings may also bypass an application's structured logs. Do not use
the presence or absence of the warning as a per-request completeness signal.

As of 0.2.27, `predict_entities` and `inference` do not provide a
`return_truncation_info` result and do not have a `truncation="error"` mode.
[Issue #231](https://github.com/urchade/GLiNER/issues/231) tracks the request for
fail-on-truncation behavior.

## Production preflight without processor internals

Use the public `prepare_batch` stage to count exactly the splitter tokens that
stateless inference will receive. This avoids depending on
`model.data_processor.words_splitter`:

```python
def truncation_info(model, text, labels):
    prepared = model.prepare_batch(text, labels)
    num_tokens = len(prepared["tokens"][0]) if prepared["tokens"] else 0
    max_len = model.config.max_len
    return {
        "truncated": num_tokens > max_len,
        "num_tokens": num_tokens,
        "max_len": max_len,
    }


info = truncation_info(model, text, labels)
if info["truncated"]:
    raise ValueError(
        f"GLiNER input has {info['num_tokens']} tokens; "
        f"the model limit is {info['max_len']}"
    )

entities = model.predict_entities(text, labels)
```

For a batch, `prepared["tokens"]` contains one token list per non-empty input;
`prepared["valid_to_orig_idx"]` maps those lists back to the original batch
indices. Emit the resulting information in the service response or reject the
request before inference, according to the service contract.

This preflight covers `config.max_len`; it does not measure a combined
uni-encoder subword sequence, an inference-packing limit, or a cached streaming
session's remaining context.

## Processing long documents

For full-document coverage, split the text into overlapping windows no longer
than `config.max_len`, predict each window, shift its character offsets back to
document coordinates, and reconcile duplicate predictions from overlaps. The
public preparation result includes the exact token-to-character maps needed to
make splitter-aligned windows:

```python
def iter_gliner_windows(model, text, labels, overlap):
    prepared = model.prepare_batch(text, labels)
    if not prepared["tokens"]:
        return

    tokens = prepared["tokens"][0]
    starts = prepared["start_token_map"][0]
    ends = prepared["end_token_map"][0]
    window_size = model.config.max_len

    if not 0 <= overlap < window_size:
        raise ValueError("overlap must be in [0, model.config.max_len)")

    step = window_size - overlap
    for first in range(0, len(tokens), step):
        last = min(first + window_size, len(tokens))
        char_start = starts[first]
        char_end = ends[last - 1]
        yield char_start, text[char_start:char_end]
        if last == len(tokens):
            break
```

For span models, an overlap of at least `max_width - 1` tokens ensures that an
entity no wider than `max_width` is fully contained in some window. A larger
overlap may be useful for contextual accuracy, and token-level models may need
a task-specific overlap because their entity length is not bounded by
`max_width`. Relation extraction also needs application-specific merging;
relations whose endpoints never occur together in a window cannot be inferred.

GLiNER does not merge window outputs for you. When shifting an entity from a
window beginning at `char_start`, add `char_start` to both `entity["start"]` and
`entity["end"]`. A common overlap policy is to group predictions by
`(start, end, label)` and retain the highest score.

## Changing `max_len`

The loader's `max_length` argument overrides the saved `config.max_len`:

```python
model = GLiNER.from_pretrained(
    "urchade/gliner_small-v2.1",
    max_length=512,
)
```

The name difference is intentional: `max_length=` at load time writes
`model.config.max_len`. Passing `max_length` or `truncation` to
`predict_entities` is not a supported way to change the word-level limit.

Increasing this value does not resize the backbone context, change the
transformer tokenizer's `model_max_length`, or guarantee quality beyond the
lengths used to train the checkpoint. Subword expansion and the label prompt
can make the combined sequence longer than the word-token count suggests, and
longer inputs require more memory. Prefer windowing unless the checkpoint and
backbone are known to support the larger value.

## Architecture-specific limits

| Architecture or path | How the label prompt is encoded | Effective-limit notes |
|---|---|---|
| UniEncoderSpan and UniEncoderToken | Prompt and text share one backbone sequence | `config.max_len` is text-only, but a finite backbone/tokenizer subword limit includes the prompt |
| UniEncoder span/token decoders | Main encoder prompt and text share a sequence; generated labels use an auxiliary decoder | The main encoder has the same two-stage limits as other uni-encoders; the decoder has its own generation limits |
| UniEncoder relation extraction | Entity labels, relation labels, and text share one sequence | Both prompt types can consume the combined backbone context, but neither changes the first-stage text-only `config.max_len` check |
| BiEncoderSpan and BiEncoderToken | Text and labels are encoded separately | Label count does not consume the text encoder sequence; each encoder still has its own tokenizer/backbone limit |
| StreamingSpan without `session_id` | Prompt and text share one causal sequence | Uses the normal stateless `config.max_len` prefix truncation, followed by the causal backbone limit |
| Cached StreamingSpan session | The initial prompt and all appended text share the decoder context | Stateless `config.max_len` truncation is bypassed; exceeding the smaller of `max_cache_length` and the decoder's native limit raises `ValueError` instead of dropping old text |

Inference packing adds another independent setting:
`InferencePackingConfig.max_length` is measured in already-tokenized backbone
token IDs, not splitter tokens. If one encoded request is longer than that
value, the current packer keeps its first `max_length` token IDs without the
`config.max_len` warning. Set the packing limit for the complete encoded
request (including a uni-encoder prompt), or disable packing for requests that
may exceed it.

Finally, `max_width` is not an input-length limit. It controls the widest
candidate entity span, in splitter tokens, for span-based architectures.
