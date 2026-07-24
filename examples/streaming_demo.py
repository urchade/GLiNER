"""Beautiful word-by-word NER demo for a trained StreamingSpan checkpoint.

Example:
    python examples/streaming_span.py models/checkpoint-15000 --inbrowser
"""

from __future__ import annotations

# This demo intentionally prints its startup URL and terminal fallback output.
# Embedded CSS is intentionally kept one selector per line in several places.
# ruff: noqa: E501, T201
import time
import argparse
from html import escape
from uuid import uuid4
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

import torch

from gliner import GLiNER

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_TEXT = "Alice Johnson joined Acme Corporation in London on Monday."
DEFAULT_LABELS = "person, organization, location, date"


def _new_session_id() -> str:
    """Return a cache key for one browser session."""
    return f"gradio-{uuid4().hex}"


EXAMPLES = [
    [
        "Alice Johnson joined Acme Corporation in London on Monday.",
        "person, organization, location, date",
    ],
    [
        "Dr. Maya Chen presented the new malaria vaccine at the World Health Organization summit in Geneva.",
        "person, job title, medicine, organization, event, location",
    ],
    [
        "Apple CEO Tim Cook met Microsoft chairman Satya Nadella in Seattle on 14 March 2026.",
        "company, person, job title, location, date",
    ],
    [
        "Serena Williams won the Australian Open in Melbourne and later founded Serena Ventures.",
        "person, competition, location, organization",
    ],
    [
        "The European Space Agency launched the JUICE spacecraft from French Guiana toward Jupiter.",
        "organization, spacecraft, location, planet",
    ],
    [
        "PyTorch and TensorFlow are popular machine-learning frameworks used with Python on Linux.",
        "software, programming language, operating system",
    ],
]

# Accent, translucent background, dark foreground.
LABEL_COLORS = [
    ("#7c3aed", "#ede9fe", "#4c1d95"),
    ("#0891b2", "#cffafe", "#164e63"),
    ("#db2777", "#fce7f3", "#831843"),
    ("#059669", "#d1fae5", "#064e3b"),
    ("#d97706", "#fef3c7", "#78350f"),
    ("#2563eb", "#dbeafe", "#1e3a8a"),
    ("#dc2626", "#fee2e2", "#7f1d1d"),
    ("#4f46e5", "#e0e7ff", "#312e81"),
]


@dataclass(frozen=True)
class StreamStep:
    """One completed word-level inference update."""

    index: int
    total: int
    token: str
    token_start: int
    token_end: int
    visible_text: str
    entities: tuple[dict[str, Any], ...]
    added: tuple[dict[str, Any], ...]
    updated: tuple[dict[str, Any], ...]
    removed: tuple[dict[str, Any], ...]


def parse_labels(value: str) -> list[str]:
    """Parse a comma-separated label list while preserving its order."""
    labels = list(dict.fromkeys(label.strip() for label in value.split(",") if label.strip()))
    if not labels:
        raise ValueError("Provide at least one label, separated by commas.")
    return labels


def iter_word_chunks(model, text: str) -> Iterator[tuple[str, str, int, int]]:
    """Yield one model-split word at a time without changing text offsets."""
    token_batches, start_batches, end_batches = model.prepare_inputs([text])
    tokens = token_batches[0]
    token_starts = start_batches[0]
    token_ends = end_batches[0]
    previous_end = 0

    for index, (token, token_start, token_end) in enumerate(zip(tokens, token_starts, token_ends)):
        # Whitespace before a token belongs to that token's chunk. Include any
        # trailing whitespace in the final chunk so concatenation recreates the
        # original text exactly.
        chunk_end = len(text) if index == len(tokens) - 1 else token_end
        yield text[previous_end:chunk_end], token, token_start, token_end
        previous_end = chunk_end


def entity_key(entity: dict[str, Any]) -> tuple[int, int, str]:
    """Return the stable identity of an entity prediction."""
    return int(entity["start"]), int(entity["end"]), str(entity["label"])


def _sorted_entities(entities) -> tuple[dict[str, Any], ...]:
    return tuple(sorted(entities, key=lambda entity: (*entity_key(entity), -float(entity["score"]))))


def stream_steps(
    model,
    text: str,
    labels: list[str],
    session_id: str,
    threshold: float,
    flat_ner: bool,
    multi_label: bool,
) -> Iterator[StreamStep]:
    """Stream words through the compatibility ``inference(session_id=...)`` API."""
    chunks = list(iter_word_chunks(model, text))
    if not chunks:
        raise ValueError("Text must contain at least one recognizable word.")

    current_entities: dict[tuple[int, int, str], dict[str, Any]] = {}
    visible_text = ""

    model.clear_session(session_id)
    try:
        for index, (chunk, token, token_start, token_end) in enumerate(chunks, start=1):
            visible_text += chunk
            predictions = model.inference(
                [chunk],
                labels,
                session_id=[session_id],
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )[0]

            previous_entities = current_entities
            step_entities = {entity_key(entity): entity for entity in predictions}
            # Session inference returns the complete active prediction snapshot.
            # Replacing it allows rolling reclassification to report removals
            # and label changes as later context arrives.
            current_entities = step_entities

            added_keys = current_entities.keys() - previous_entities.keys()
            removed_keys = previous_entities.keys() - current_entities.keys()
            updated_keys = {
                key
                for key in current_entities.keys() & previous_entities.keys()
                if float(current_entities[key]["score"]) != float(previous_entities[key]["score"])
            }

            yield StreamStep(
                index=index,
                total=len(chunks),
                token=token,
                token_start=token_start,
                token_end=token_end,
                visible_text=visible_text,
                entities=_sorted_entities(current_entities.values()),
                added=_sorted_entities(current_entities[key] for key in added_keys),
                updated=_sorted_entities(current_entities[key] for key in updated_keys),
                removed=_sorted_entities(previous_entities[key] for key in removed_keys),
            )
    finally:
        model.clear_session(session_id)


def _label_color(label: str, labels: list[str]) -> tuple[str, str, str]:
    try:
        index = labels.index(label)
    except ValueError:
        index = sum(label.encode("utf-8"))
    return LABEL_COLORS[index % len(LABEL_COLORS)]


def span_context_label(model) -> str:
    """Return a user-facing name for the span-context strategy."""
    return "bidirectional" if model.model.span_rep_layer.uses_bidirectional_context else "causal"


def _label_legend(labels: list[str]) -> str:
    chips = []
    for label in labels:
        accent, background, foreground = _label_color(label, labels)
        chips.append(
            '<span class="label-chip" '
            f'style="--label-accent:{accent};--label-bg:{background};--label-fg:{foreground}">'
            f"<i></i>{escape(label)}</span>"
        )
    return '<div class="label-legend">' + "".join(chips) + "</div>"


def render_annotated_text(
    text: str,
    entities: tuple[dict[str, Any], ...],
    labels: list[str],
    current_word: tuple[int, int] | None = None,
) -> str:
    """Render streamed text with safe, overlap-aware entity highlighting."""
    if not text:
        body = (
            '<div class="empty-stage"><span class="empty-orb">✦</span>'
            "Press <strong>Start streaming</strong> to watch entities emerge.</div>"
        )
        return f'<section class="visual-card">{_label_legend(labels)}{body}</section>'

    valid_entities = tuple(entity for entity in entities if 0 <= int(entity["start"]) < int(entity["end"]) <= len(text))
    boundaries = {0, len(text)}
    for entity in valid_entities:
        boundaries.update((int(entity["start"]), int(entity["end"])))
    if current_word is not None:
        boundaries.update((max(0, current_word[0]), min(len(text), current_word[1])))
    ordered_boundaries = sorted(boundary for boundary in boundaries if 0 <= boundary <= len(text))

    fragments = []
    for start, end in zip(ordered_boundaries, ordered_boundaries[1:]):
        if start == end:
            continue
        active = [entity for entity in valid_entities if int(entity["start"]) < end and int(entity["end"]) > start]
        is_current = current_word is not None and current_word[0] < end and current_word[1] > start
        classes = ["text-fragment"]
        styles = []
        title = ""

        if active:
            active.sort(key=lambda entity: float(entity["score"]), reverse=True)
            colors = [_label_color(str(entity["label"]), labels) for entity in active[:2]]
            classes.append("entity-fragment")
            styles.append(f"border-color:{colors[0][0]}")
            if len(colors) == 1:
                styles.append(f"background:{colors[0][1]}")
                styles.append(f"color:{colors[0][2]}")
            else:
                styles.append(f"background:linear-gradient(135deg,{colors[0][1]} 0 50%,{colors[1][1]} 50% 100%)")
                styles.append(f"color:{colors[0][2]}")
            title = "; ".join(f"{entity['label']} · {float(entity['score']):.3f}" for entity in active)
        if is_current:
            classes.append("current-word")

        style_attr = f' style="{";".join(styles)}"' if styles else ""
        title_attr = f' title="{escape(title, quote=True)}"' if title else ""
        fragments.append(f'<span class="{" ".join(classes)}"{style_attr}{title_attr}>{escape(text[start:end])}</span>')

        ending_here = [entity for entity in valid_entities if int(entity["end"]) == end]
        for entity in sorted(ending_here, key=lambda item: (int(item["start"]), str(item["label"]))):
            accent, background, foreground = _label_color(str(entity["label"]), labels)
            fragments.append(
                '<span class="inline-label" '
                f'style="--label-accent:{accent};--label-bg:{background};--label-fg:{foreground}">'
                f"{escape(str(entity['label']))}<small>{float(entity['score']):.2f}</small></span>"
            )

    return (
        '<section class="visual-card">'
        '<div class="card-kicker"><span class="live-dot"></span>Live recognition</div>'
        f"{_label_legend(labels)}"
        f'<div class="stream-text">{"".join(fragments)}<span class="stream-caret"></span></div>'
        "</section>"
    )


def render_word_rail(tokens: list[str], current_index: int = -1) -> str:
    """Render all model-split words as a processed/current/pending timeline."""
    word_chips = []
    for index, token in enumerate(tokens):
        if index < current_index:
            state = "processed"
        elif index == current_index:
            state = "active"
        else:
            state = "pending"
        word_chips.append(f'<span class="word-chip {state}"><small>{index + 1:02d}</small>{escape(token)}</span>')
    return (
        '<section class="word-rail-card"><div class="rail-heading">'
        "<span>Model word stream</span><small>Every chip is one inference step</small></div>"
        f'<div class="word-rail">{"".join(word_chips)}</div></section>'
    )


def render_entity_cards(entities: tuple[dict[str, Any], ...], labels: list[str]) -> str:
    """Render the current entity set as confidence cards."""
    if not entities:
        cards = (
            '<div class="no-entities"><span>⌁</span><div><strong>No entities yet</strong>'
            "<small>The panel updates as soon as a span crosses the threshold.</small></div></div>"
        )
    else:
        rendered = []
        for entity in entities:
            label = str(entity["label"])
            accent, background, foreground = _label_color(label, labels)
            score = max(0.0, min(1.0, float(entity["score"])))
            rendered.append(
                '<article class="entity-card" '
                f'style="--label-accent:{accent};--label-bg:{background};--label-fg:{foreground}">'
                '<div class="entity-card-top">'
                f'<span class="entity-type">{escape(label)}</span><strong>{score:.1%}</strong></div>'
                f'<div class="entity-value">{escape(str(entity["text"]))}</div>'
                f'<div class="entity-offset">characters {int(entity["start"])}&ndash;{int(entity["end"])}</div>'
                f'<div class="confidence-track"><i style="width:{score * 100:.1f}%"></i></div>'
                "</article>"
            )
        cards = '<div class="entity-grid">' + "".join(rendered) + "</div>"

    return (
        '<section class="entities-section"><div class="section-heading">'
        '<div><span class="eyebrow">Structured output</span><h3>Recognized entities</h3></div>'
        f'<span class="entity-count">{len(entities)}</span></div>{cards}</section>'
    )


def render_status(
    step: int,
    total: int,
    entity_count: int,
    mode: str,
    state: str = "ready",
    message: str | None = None,
) -> str:
    """Render progress and cache state for the current stream."""
    progress = 0 if total == 0 else min(100, round(step / total * 100))
    state_label = {
        "ready": "Ready to stream",
        "streaming": "Streaming",
        "complete": "Complete",
        "stopped": "Stopped",
        "error": "Could not continue",
    }[state]
    prompt_label = "Prompt pending" if step == 0 else "Prompt cached"
    extra = f'<div class="status-message">{escape(message)}</div>' if message else ""
    return (
        f'<section class="status-card {state}">'
        '<div class="status-top"><div class="status-title">'
        f'<span class="status-icon"></span><strong>{state_label}</strong></div>'
        f'<span class="status-percent">{progress}%</span></div>'
        f'<div class="progress-track"><i style="width:{progress}%"></i></div>'
        '<div class="metric-row">'
        f"<span><strong>{step}</strong><small>of {total} words</small></span>"
        f"<span><strong>{entity_count}</strong><small>entities</small></span>"
        f"<span><strong>{escape(mode)}</strong><small>span context</small></span>"
        f"<span><strong>{prompt_label}</strong><small>session KV cache</small></span>"
        f"</div>{extra}</section>"
    )


def initial_view(model, text: str, labels_value: str):
    """Build the four UI panels before inference starts."""
    labels = parse_labels(labels_value)
    chunks = list(iter_word_chunks(model, text))
    if not chunks:
        raise ValueError("Text must contain at least one recognizable word.")
    tokens = [token for _, token, _, _ in chunks]
    mode = span_context_label(model)
    return (
        render_status(0, len(tokens), 0, mode),
        render_annotated_text("", (), labels),
        render_word_rail(tokens),
        render_entity_cards((), labels),
    )


def create_streaming_handler(model):
    """Create a Gradio generator bound to one loaded model."""

    def recognize(
        text: str,
        labels_value: str,
        threshold: float,
        delay: float,
        nested_ner: bool,
        multi_label: bool,
        session_id: str,
    ):
        labels: list[str] = []
        tokens: list[str] = []
        mode = span_context_label(model)
        last_step: StreamStep | None = None
        try:
            labels = parse_labels(labels_value)
            chunks = list(iter_word_chunks(model, text))
            if not chunks:
                raise ValueError("Text must contain at least one recognizable word.")
            tokens = [token for _, token, _, _ in chunks]

            yield (
                render_status(0, len(tokens), 0, mode),
                render_annotated_text("", (), labels),
                render_word_rail(tokens),
                render_entity_cards((), labels),
            )

            for step in stream_steps(
                model,
                text,
                labels,
                session_id,
                threshold,
                flat_ner=not nested_ner,
                multi_label=multi_label,
            ):
                last_step = step
                state = "complete" if step.index == step.total else "streaming"
                yield (
                    render_status(step.index, step.total, len(step.entities), mode, state=state),
                    render_annotated_text(
                        step.visible_text,
                        step.entities,
                        labels,
                        current_word=(step.token_start, step.token_end),
                    ),
                    render_word_rail(tokens, step.index - 1),
                    render_entity_cards(step.entities, labels),
                )
                if delay and step.index != step.total:
                    time.sleep(delay)
        except Exception as error:
            total = len(tokens)
            completed = last_step.index if last_step is not None else 0
            entities = last_step.entities if last_step is not None else ()
            visible_text = last_step.visible_text if last_step is not None else ""
            current_word = (last_step.token_start, last_step.token_end) if last_step is not None else None
            safe_labels = labels or ["entity"]
            yield (
                render_status(completed, total, len(entities), mode, state="error", message=str(error)),
                render_annotated_text(visible_text, entities, safe_labels, current_word=current_word),
                render_word_rail(tokens, completed - 1),
                render_entity_cards(entities, safe_labels),
            )
        finally:
            model.clear_session(session_id)

    return recognize


def stream_words(
    model,
    text: str,
    labels: list[str],
    session_id: str,
    threshold: float,
    delay: float,
    flat_ner: bool,
    multi_label: bool,
) -> list[dict[str, Any]]:
    """Terminal wrapper retained for quick smoke tests and scripting."""
    final_entities: tuple[dict[str, Any], ...] = ()
    for step in stream_steps(model, text, labels, session_id, threshold, flat_ner, multi_label):
        final_entities = step.entities
        print(f"[{step.index:02d}/{step.total:02d}] {step.token!r}")
        for entity in (*step.added, *step.updated):
            print(
                f"  {entity['text']!r} -> {entity['label']} "
                f"[{entity['start']}:{entity['end']}] score={float(entity['score']):.3f}"
            )
        if not step.added and not step.updated and not step.removed:
            print("  no entity change")
        print(flush=True)
        if delay and step.index != step.total:
            time.sleep(delay)
    return list(final_entities)


HERO_HTML = """
<section class="demo-hero">
  <div class="hero-copy">
    <div class="hero-badge"><span></span>StreamingSpan live session</div>
    <h1>Watch entities <em>emerge</em><br>one word at a time.</h1>
    <p>GLiNER reads a token, extends its KV cache, revisits provisional spans with
       bounded future context, and updates recognized entities as they evolve.</p>
    <div class="hero-pills">
      <span>⚡ Cached decoding</span><span>✦ Open labels</span><span>↗ Exact offsets</span>
    </div>
  </div>
  <div class="hero-visual" aria-hidden="true">
    <div class="orbit orbit-one"></div><div class="orbit orbit-two"></div>
    <div class="core-mark">GL<span>i</span>NER<small>STREAM</small></div>
    <div class="floating-tag tag-one">PERSON</div>
    <div class="floating-tag tag-two">LOCATION</div>
    <div class="floating-tag tag-three">ORG</div>
  </div>
</section>
"""


DEMO_CSS = """
.gradio-container {
  max-width: 1480px !important;
  background:
    radial-gradient(circle at 12% -10%, color-mix(in srgb, var(--primary-400) 18%, transparent), transparent 34rem),
    radial-gradient(circle at 92% 18%, color-mix(in srgb, #06b6d4 13%, transparent), transparent 30rem),
    var(--body-background-fill);
}
.demo-hero {
  position: relative; overflow: hidden; min-height: 330px; margin: 16px 0 24px; padding: 44px 52px;
  display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(280px, .6fr); align-items: center;
  border: 1px solid color-mix(in srgb, var(--primary-400) 24%, var(--border-color-primary));
  border-radius: 30px; color: #f8fafc;
  background: linear-gradient(125deg, #111827 0%, #312e81 52%, #0e7490 118%);
  box-shadow: 0 28px 80px rgba(15, 23, 42, .22);
}
.demo-hero:before { content:""; position:absolute; inset:0; opacity:.28; pointer-events:none;
  background-image: linear-gradient(rgba(255,255,255,.08) 1px,transparent 1px),
                    linear-gradient(90deg,rgba(255,255,255,.08) 1px,transparent 1px);
  background-size: 38px 38px; mask-image:linear-gradient(to right,black,transparent 75%); }
.hero-copy { position:relative; z-index:2; max-width:790px; }
.hero-badge { display:inline-flex; align-items:center; gap:9px; padding:7px 12px; border:1px solid rgba(255,255,255,.2);
  border-radius:999px; background:rgba(255,255,255,.09); font-size:12px; font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
.hero-badge span,.live-dot { width:8px; height:8px; border-radius:50%; background:#5eead4; box-shadow:0 0 0 5px rgba(94,234,212,.13); animation:live-pulse 1.7s infinite; }
.demo-hero h1 { margin:18px 0 14px; color:white; font-size:clamp(38px,5vw,70px); line-height:.98; letter-spacing:-.055em; }
.demo-hero h1 em { color:#a5f3fc; font-style:normal; }
.demo-hero p { max-width:720px; margin:0; color:#cbd5e1; font-size:17px; line-height:1.65; }
.hero-pills { display:flex; flex-wrap:wrap; gap:9px; margin-top:24px; }
.hero-pills span { padding:8px 12px; border-radius:12px; background:rgba(255,255,255,.08); color:#e2e8f0; font-size:13px; }
.hero-visual { position:relative; z-index:1; min-height:240px; display:grid; place-items:center; }
.core-mark { width:154px; height:154px; display:grid; place-content:center; text-align:center; border-radius:42px;
  background:linear-gradient(145deg,rgba(255,255,255,.22),rgba(255,255,255,.06)); border:1px solid rgba(255,255,255,.32);
  box-shadow:inset 0 1px rgba(255,255,255,.35),0 24px 50px rgba(0,0,0,.24); backdrop-filter:blur(12px);
  font-size:29px; font-weight:850; letter-spacing:-.06em; transform:rotate(-3deg); }
.core-mark span { color:#67e8f9; }.core-mark small { display:block; margin-top:5px; font-size:9px; letter-spacing:.34em; color:#a5f3fc; }
.orbit { position:absolute; border:1px solid rgba(255,255,255,.2); border-radius:50%; }
.orbit-one { width:240px;height:240px;animation:spin 15s linear infinite; }.orbit-two { width:310px;height:150px;transform:rotate(-25deg); }
.floating-tag { position:absolute; padding:7px 10px; border-radius:8px; background:white; color:#312e81; font-size:10px; font-weight:850;
  box-shadow:0 12px 25px rgba(0,0,0,.2); letter-spacing:.08em; }
.tag-one { top:23px;right:16px;transform:rotate(7deg); }.tag-two { bottom:24px;left:5px;transform:rotate(-8deg); }.tag-three { right:0;bottom:52px;transform:rotate(4deg); }
.glass-panel { padding:20px !important; border:1px solid var(--border-color-primary) !important; border-radius:22px !important;
  background:color-mix(in srgb,var(--block-background-fill) 92%,transparent) !important; box-shadow:0 15px 45px rgba(15,23,42,.07); }
#control-panel textarea { font-size:15px; line-height:1.55; }
#start-stream { min-height:48px; border:0; background:linear-gradient(110deg,#7c3aed,#2563eb,#0891b2); box-shadow:0 12px 24px rgba(79,70,229,.22); }
#start-stream:hover { transform:translateY(-1px); filter:saturate(1.12); }
.status-card,.visual-card,.word-rail-card,.entities-section { border:1px solid var(--border-color-primary); border-radius:22px;
  background:color-mix(in srgb,var(--block-background-fill) 94%,transparent); box-shadow:0 14px 38px rgba(15,23,42,.065); }
.status-card { padding:18px 20px; margin-bottom:14px; }
.status-top,.status-title,.metric-row,.section-heading,.entity-card-top,.rail-heading { display:flex; align-items:center; }
.status-top,.section-heading,.entity-card-top,.rail-heading { justify-content:space-between; }
.status-title { gap:10px; }.status-icon { width:10px;height:10px;border-radius:50%;background:#94a3b8; }
.status-card.streaming .status-icon { background:#22c55e;box-shadow:0 0 0 6px rgba(34,197,94,.12);animation:live-pulse 1.6s infinite; }
.status-card.complete .status-icon { background:#06b6d4; }.status-card.error .status-icon { background:#ef4444; }
.status-percent { font-size:13px;font-weight:800;color:var(--primary-500); }
.progress-track,.confidence-track { overflow:hidden;background:color-mix(in srgb,var(--body-text-color) 9%,transparent);border-radius:999px; }
.progress-track { height:7px;margin:13px 0 15px; }.progress-track i { display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#7c3aed,#06b6d4);transition:width .35s ease; }
.metric-row { gap:9px;display:grid;grid-template-columns:repeat(4,1fr); }.metric-row span { padding:9px 11px;border-radius:12px;background:var(--background-fill-secondary); }
.metric-row strong,.metric-row small { display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }.metric-row strong { font-size:12px; }.metric-row small { margin-top:2px;color:var(--body-text-color-subdued);font-size:10px; }
.status-message { margin-top:12px;padding:10px 12px;border-radius:10px;color:#991b1b;background:#fee2e2;font-size:12px; }
.visual-card { min-height:225px;padding:22px 24px; }.card-kicker { display:flex;align-items:center;gap:10px;margin-bottom:14px;color:var(--body-text-color-subdued);font-size:11px;font-weight:800;letter-spacing:.09em;text-transform:uppercase; }
.label-legend { display:flex;flex-wrap:wrap;gap:7px;margin-bottom:18px; }.label-chip { display:inline-flex;align-items:center;gap:7px;padding:6px 9px;border-radius:999px;background:var(--label-bg);color:var(--label-fg);font-size:11px;font-weight:750; }
.label-chip i { width:7px;height:7px;border-radius:50%;background:var(--label-accent); }
.stream-text { min-height:95px;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--body-text-color);font-size:clamp(20px,2.1vw,29px);font-weight:560;line-height:2.05;letter-spacing:-.018em; }
.text-fragment { border-radius:5px;transition:background .25s ease; }.entity-fragment { padding:2px 1px;border-bottom:3px solid;box-decoration-break:clone;-webkit-box-decoration-break:clone; }
.current-word { position:relative;outline:2px solid color-mix(in srgb,var(--primary-400) 70%,transparent);outline-offset:3px;animation:word-pop .45s ease both; }
.inline-label { display:inline-flex;align-items:center;gap:4px;margin:0 4px 0 2px;padding:3px 6px;transform:translateY(-9px);border:1px solid color-mix(in srgb,var(--label-accent) 28%,transparent);
  border-radius:7px;background:var(--label-bg);color:var(--label-fg);font-size:9px;font-weight:850;line-height:1;letter-spacing:.04em;text-transform:uppercase;white-space:nowrap; }
.inline-label small { padding-left:4px;border-left:1px solid color-mix(in srgb,var(--label-accent) 25%,transparent);font-size:8px; }
.stream-caret { display:inline-block;width:2px;height:1.15em;margin-left:3px;vertical-align:-.14em;background:var(--primary-500);animation:blink 1s step-end infinite; }
.empty-stage { min-height:125px;display:grid;place-content:center;justify-items:center;gap:10px;color:var(--body-text-color-subdued);text-align:center; }.empty-orb { display:grid;place-items:center;width:42px;height:42px;border-radius:14px;color:white;background:linear-gradient(135deg,#7c3aed,#0891b2);box-shadow:0 9px 22px rgba(79,70,229,.2); }
.word-rail-card { margin-top:14px;padding:17px 19px; }.rail-heading span { font-size:12px;font-weight:800; }.rail-heading small { color:var(--body-text-color-subdued);font-size:10px; }
.word-rail { display:flex;gap:7px;margin-top:12px;padding:4px 2px 8px;overflow-x:auto;scrollbar-width:thin; }.word-chip { flex:none;display:inline-flex;align-items:center;gap:6px;padding:7px 10px;border:1px solid var(--border-color-primary);border-radius:10px;color:var(--body-text-color-subdued);background:var(--background-fill-secondary);font-size:12px;transition:.3s ease; }
.word-chip small { font-size:8px;opacity:.55; }.word-chip.processed { color:var(--body-text-color);border-color:color-mix(in srgb,#10b981 28%,var(--border-color-primary));background:color-mix(in srgb,#10b981 8%,var(--block-background-fill)); }
.word-chip.active { color:white;border-color:transparent;background:linear-gradient(115deg,#7c3aed,#2563eb);box-shadow:0 8px 18px rgba(79,70,229,.25);transform:translateY(-2px) scale(1.03); }
.entities-section { margin-top:20px;padding:23px; }.section-heading { margin-bottom:16px; }.section-heading h3 { margin:2px 0 0;font-size:21px; }.eyebrow { color:var(--primary-500);font-size:9px;font-weight:850;letter-spacing:.12em;text-transform:uppercase; }
.entity-count { display:grid;place-items:center;min-width:38px;height:38px;border-radius:12px;color:white;background:linear-gradient(135deg,#7c3aed,#0891b2);font-weight:850; }
.entity-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:11px; }.entity-card { position:relative;overflow:hidden;padding:15px;border:1px solid color-mix(in srgb,var(--label-accent) 22%,var(--border-color-primary));border-radius:15px;background:linear-gradient(145deg,var(--label-bg),var(--block-background-fill) 72%); }
.entity-card:before { content:"";position:absolute;inset:0 auto 0 0;width:4px;background:var(--label-accent); }.entity-type { padding:4px 7px;border-radius:7px;background:var(--label-bg);color:var(--label-fg);font-size:9px;font-weight:850;letter-spacing:.06em;text-transform:uppercase; }
.entity-card-top strong { color:var(--label-accent);font-size:12px; }.entity-value { margin-top:13px;font-size:16px;font-weight:780;line-height:1.25; }.entity-offset { margin:5px 0 12px;color:var(--body-text-color-subdued);font-size:9px; }.confidence-track { height:5px; }.confidence-track i { display:block;height:100%;border-radius:inherit;background:var(--label-accent); }
.no-entities { display:flex;align-items:center;justify-content:center;gap:14px;min-height:100px;border:1px dashed var(--border-color-primary);border-radius:15px;color:var(--body-text-color-subdued); }.no-entities>span { font-size:34px; }.no-entities strong,.no-entities small { display:block; }.no-entities small { margin-top:4px;font-size:11px; }
.examples-heading { margin:30px 0 8px;text-align:center; }.examples-heading h2 { margin-bottom:4px;font-size:26px; }.examples-heading p { color:var(--body-text-color-subdued); }
@keyframes live-pulse { 50% { opacity:.55;transform:scale(.82); } } @keyframes blink { 50% { opacity:0; } }
@keyframes spin { to { transform:rotate(360deg); } } @keyframes word-pop { from { opacity:.45;transform:translateY(4px); } }
@media (max-width:850px) { .demo-hero { grid-template-columns:1fr;padding:32px 25px; }.hero-visual { display:none; }.metric-row { grid-template-columns:repeat(2,1fr); }.glass-panel { padding:14px !important; } }
"""


def require_gradio():
    """Import Gradio lazily so non-UI helpers remain usable without it."""
    try:
        import gradio as gr  # noqa: PLC0415
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("Install Gradio to run this demo: pip install 'gradio>=6'") from error
    return gr


def build_demo(
    model,
    model_source: str,
    default_text: str = DEFAULT_TEXT,
    default_labels: str = DEFAULT_LABELS,
    default_threshold: float = 0.5,
    default_delay: float = 0.3,
):
    """Construct the Gradio Blocks application."""
    gr = require_gradio()

    initial_status, initial_text, initial_words, initial_entities = initial_view(
        model,
        default_text,
        default_labels,
    )
    handler = create_streaming_handler(model)
    mode = span_context_label(model)
    model_caption = escape(model_source)

    with gr.Blocks(title="GLiNER Live — Streaming NER", fill_width=True) as demo:
        session_id = gr.State(
            value=_new_session_id(),
            time_to_live=3600,
            delete_callback=model.clear_session,
        )

        gr.HTML(HERO_HTML, container=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_id="control-panel", elem_classes="glass-panel"):
                gr.Markdown(
                    f"### Compose the stream\nLoaded `{model_caption}` · `{mode}` span context",
                    container=False,
                )
                text_input = gr.Textbox(
                    value=default_text,
                    label="Text",
                    info="The model's own word splitter determines each streaming step.",
                    lines=7,
                    max_lines=12,
                    placeholder="Paste a sentence or paragraph…",
                )
                labels_input = gr.Textbox(
                    value=default_labels,
                    label="Entity labels",
                    info="Comma-separated; labels can be anything your model learned to recognize.",
                    lines=2,
                    placeholder="person, organization, location, date",
                )
                with gr.Row():
                    threshold_input = gr.Slider(
                        0.0,
                        1.0,
                        value=default_threshold,
                        step=0.01,
                        label="Confidence threshold",
                    )
                    delay_input = gr.Slider(
                        0.0,
                        1.0,
                        value=default_delay,
                        step=0.05,
                        label="Seconds per word",
                    )
                with gr.Accordion("Recognition options", open=False):
                    nested_input = gr.Checkbox(label="Allow nested entities", value=False)
                    multi_label_input = gr.Checkbox(label="Allow multiple labels per span", value=False)
                with gr.Row():
                    start_button = gr.Button("▶  Start streaming", variant="primary", elem_id="start-stream", scale=3)
                    stop_button = gr.Button("■  Stop", variant="stop", scale=1)
                    reset_button = gr.Button("↺", variant="secondary", scale=0, min_width=52)

            with gr.Column(scale=7):
                status_output = gr.HTML(initial_status, container=False, elem_id="stream-status")
                text_output = gr.HTML(initial_text, container=False, elem_id="stream-visual")
                words_output = gr.HTML(initial_words, container=False, elem_id="word-timeline")

        entities_output = gr.HTML(initial_entities, container=False, elem_id="entity-results")

        gr.HTML(
            '<div class="examples-heading"><h2>Start with a story</h2>'
            "<p>Pick an example, adjust its open label vocabulary, then start the stream.</p></div>",
            container=False,
        )
        gr.Examples(
            examples=EXAMPLES,
            inputs=[text_input, labels_input],
            example_labels=["Work", "Medicine", "Technology", "Sports", "Space", "Software"],
            label="Curated examples",
            examples_per_page=6,
        )

        stream_inputs = [
            text_input,
            labels_input,
            threshold_input,
            delay_input,
            nested_input,
            multi_label_input,
            session_id,
        ]
        stream_outputs = [status_output, text_output, words_output, entities_output]
        run_event = start_button.click(
            fn=handler,
            inputs=stream_inputs,
            outputs=stream_outputs,
            show_progress="hidden",
            concurrency_limit=1,
            concurrency_id="streaming-span-inference",
            stream_every=0.05,
        )

        def stop_stream(active_session_id: str):
            model.clear_session(active_session_id)
            return render_status(0, 0, 0, mode, state="stopped", message="The session cache was cleared.")

        stop_button.click(
            fn=stop_stream,
            inputs=session_id,
            outputs=status_output,
            cancels=[run_event],
            queue=False,
        )

        def reset_stream(active_session_id: str, text: str, labels_value: str):
            model.clear_session(active_session_id)
            try:
                return initial_view(model, text, labels_value)
            except ValueError as error:
                return (
                    render_status(0, 0, 0, mode, state="error", message=str(error)),
                    render_annotated_text("", (), ["entity"]),
                    render_word_rail([]),
                    render_entity_cards((), ["entity"]),
                )

        reset_button.click(
            fn=reset_stream,
            inputs=[session_id, text_input, labels_input],
            outputs=stream_outputs,
            cancels=[run_event],
            queue=False,
        )

        # Use an explicit load event rather than a callable State value. Gradio
        # 6.20.0 leaves callable State values unevaluated, so callbacks otherwise
        # receive the factory function instead of the generated cache key.
        demo.load(fn=_new_session_id, outputs=session_id, queue=False)

    return demo


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` to the best available PyTorch device."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local path or Hugging Face ID of a trained StreamingSpan checkpoint")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Initial text shown in the demo")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Initial comma-separated labels")
    parser.add_argument("--threshold", type=float, default=0.5, help="Initial confidence threshold")
    parser.add_argument("--delay", type=float, default=0.3, help="Initial seconds-per-word playback speed")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model files")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio server bind address")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--inbrowser", action="store_true", help="Open the demo in a browser")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if not 0.0 <= args.delay <= 1.0:
        raise SystemExit("--delay must be between 0 and 1")
    try:
        parse_labels(args.labels)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    model = GLiNER.from_pretrained(
        args.model,
        load_tokenizer=True,
        local_files_only=args.local_files_only,
    )
    if getattr(model.config, "model_type", None) != "gliner_streaming_span":
        raise SystemExit("The checkpoint must use model_type='gliner_streaming_span'")

    device = resolve_device(args.device)
    model = model.to(device).eval()
    print(f"Loaded {args.model!r} on {device}; starting the streaming Gradio demo…")

    try:
        demo = build_demo(
            model,
            args.model,
            default_text=args.text,
            default_labels=args.labels,
            default_threshold=args.threshold,
            default_delay=args.delay,
        )
    except ModuleNotFoundError as error:
        raise SystemExit(str(error)) from error

    gr = require_gradio()
    theme = gr.themes.Soft(primary_hue="violet", secondary_hue="cyan", neutral_hue="slate")
    demo.queue(max_size=32, default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_error=True,
        theme=theme,
        css=DEMO_CSS,
    )


if __name__ == "__main__":
    main()
