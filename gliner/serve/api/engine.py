"""Plan model rows for process requests, run them, and assemble results.

Every task becomes one model row: the rendered input with the task's labels,
thresholds and decoding flags. GLiNER attends over the labels together with
the text, so tasks with different label sets cannot share a row without
changing each other's predictions; they share the *batch* instead. All rows
of a request -- and of every request batched with it -- go through the same
dynamic batcher, so a request with five tasks costs five rows of one forward
pass, not five forward passes.

Identical rows (same text, labels, thresholds, flags and adapter) are run
once, within a request and across the items of a batch.

An input longer than the model window runs over overlapping windows; spans
and relations are mapped back to the full input and merged.
"""

from __future__ import annotations

import re
import time
import asyncio
import logging
from typing import TYPE_CHECKING, Any
from dataclasses import field, dataclass

from .errors import APIError, invalid
from .request import BATCH_KEYS, TaskSpec, ParsedRequest, RequestParser, merge_options

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence, Awaitable

    from .labels import Label
    from .render import RenderedInput
    from .capabilities import Capabilities

logger = logging.getLogger(__name__)

Row = dict[str, Any]
RowOutput = dict[str, Any]

_WORD_RE = re.compile(r"\S+")


def _split_whitespace(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _WORD_RE.finditer(text)]


def _prompt_words(labels: Sequence[Label] | None) -> int:
    """Words the labels add to a row: their text plus one marker token each."""
    return sum(len(label.text.split()) + 1 for label in labels or ())


def _model_labels(labels: Sequence[Label]) -> list[str] | dict[str, str]:
    """Labels as the model takes them: plain IDs, or ``{id: text}`` when they differ."""
    if all(label.text == label.id for label in labels):
        return [label.id for label in labels]
    return {label.id: label.text for label in labels}


def _label_lookup(labels: Sequence[Label] | None) -> dict[str, str]:
    """Map what the model may return for a label (its ID or text) to the ID."""
    lookup: dict[str, str] = {}
    for label in labels or ():
        lookup[label.text] = label.id
    for label in labels or ():
        lookup[label.id] = label.id
    return lookup


def _overlaps(start: int, end: int, other: dict[str, Any]) -> bool:
    return start < other["_end"] and other["_start"] < end


def _greedy_flat(spans: list[dict[str, Any]], *, multi_label: bool) -> list[dict[str, Any]]:
    """Keep the highest-scoring spans that do not overlap.

    With ``multi_label`` the same span may still carry several labels.
    """
    kept: list[dict[str, Any]] = []
    for span in sorted(spans, key=lambda item: -item["score"]):
        start, end = span["_start"], span["_end"]
        if all(
            not _overlaps(start, end, other) or (multi_label and (start, end) == (other["_start"], other["_end"]))
            for other in kept
        ):
            kept.append(span)
    return kept


@dataclass
class RowRef:
    """One model row a task runs on."""

    task: TaskSpec
    row: Row
    offset: int  # char offset of the row text in the rendered input
    words: int  # text + prompt words, for usage reporting
    slot: int = -1  # index of the row among the executed unique rows


@dataclass
class Plan:
    parsed: ParsedRequest
    refs: list[RowRef] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)
    outputs: list[RowOutput] | None = None
    error: APIError | None = None


class ProcessEngine:
    """Validate, plan, run and assemble process requests for GLiNER.

    The engine does not run the model. ``process*`` take a row runner:

    * sync: ``run_rows(rows) -> outputs``, one output per row;
    * async: ``run_row(row) -> awaitable output``, called once per row so a
      dynamic batcher can merge rows from many requests.

    A row is ``{"text", "labels", "relations", "threshold",
    "relation_threshold", "flat_ner", "multi_label", "adapter_id"}`` and its
    output is ``{"entities": [...], "relations": [...]?}`` with character
    offsets into the row text, as ``GLiNERServer`` produces.
    """

    def __init__(
        self,
        capabilities: Capabilities,
        *,
        max_words: int,
        split_words: Callable[[str], list[tuple[int, int]]] | None = None,
        resolve_adapter: Callable[[str | None], Any] | None = None,
        window_overlap: float = 0.25,
    ) -> None:
        if not 0.0 <= window_overlap < 1.0:
            raise ValueError("window_overlap must be in [0, 1)")
        self.capabilities = capabilities
        self.parser = RequestParser(capabilities)
        self.max_words = max_words
        self.split_words = split_words or _split_whitespace
        self.resolve_adapter = resolve_adapter or (lambda adapter_id: adapter_id)
        self.window_overlap = window_overlap

    # -- public: sync -----------------------------------------------------------

    def process(self, body: Any, run_rows: Callable[[list[Row]], list[RowOutput]]) -> dict[str, Any]:
        """One request; raises ``APIError`` when it is rejected or fails."""
        outcome = self.process_many([body], run_rows)[0]
        if isinstance(outcome, APIError):
            raise outcome
        return outcome

    def process_many(
        self,
        bodies: Sequence[Any],
        run_rows: Callable[[list[Row]], list[RowOutput]],
        *,
        check_model: bool = True,
    ) -> list[dict[str, Any] | APIError]:
        """Independent requests run together; each succeeds or fails alone."""
        outcomes, plans = self._prepare(bodies, check_model=check_model)
        rows = self._schedule(plans)
        if rows:
            try:
                results: list[Any] = list(run_rows(rows))
                if len(results) != len(rows):
                    raise RuntimeError(f"Row runner returned {len(results)} outputs for {len(rows)} rows")
            except Exception as exc:
                results = [exc] * len(rows)
            self._distribute(plans, results)
        return self._finish(outcomes, plans)

    def process_batch(self, body: Any, run_rows: Callable[[list[Row]], list[RowOutput]]) -> dict[str, Any]:
        """``/v1/process/batch``: many items, per-item errors, order kept."""
        items, model = self.parse_batch(body)
        outcomes = self.process_many([item for _, item in items], run_rows, check_model=False)
        return self.batch_response(items, model, outcomes)

    # -- public: async ----------------------------------------------------------

    async def process_async(self, body: Any, run_row: Callable[[Row], Awaitable[RowOutput]]) -> dict[str, Any]:
        outcome = (await self.process_many_async([body], run_row))[0]
        if isinstance(outcome, APIError):
            raise outcome
        return outcome

    async def process_many_async(
        self,
        bodies: Sequence[Any],
        run_row: Callable[[Row], Awaitable[RowOutput]],
        *,
        check_model: bool = True,
    ) -> list[dict[str, Any] | APIError]:
        outcomes, plans = self._prepare(bodies, check_model=check_model)
        rows = self._schedule(plans)
        if rows:
            results = await asyncio.gather(*(run_row(row) for row in rows), return_exceptions=True)
            self._distribute(plans, list(results))
        return self._finish(outcomes, plans)

    async def process_batch_async(self, body: Any, run_row: Callable[[Row], Awaitable[RowOutput]]) -> dict[str, Any]:
        items, model = self.parse_batch(body)
        outcomes = await self.process_many_async([item for _, item in items], run_row, check_model=False)
        return self.batch_response(items, model, outcomes)

    # -- batch envelope -----------------------------------------------------------

    def parse_batch(self, body: Any) -> tuple[list[tuple[str | None, Any]], str]:
        if not isinstance(body, dict):
            raise invalid("The request body must be a JSON object", "")
        for key in body:
            if key not in BATCH_KEYS:
                raise invalid(f"Unknown batch field '{key}'", f"/{key}")
        model = self.parser.check_model(body.get("model"))
        raw_items = body.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise invalid("'items' must be a non-empty list", "/items")
        limit = self.capabilities.limit("max_batch_items")
        if limit and len(raw_items) > limit:
            raise invalid(f"At most {limit} items are allowed per batch", "/items")
        batch_shared = {}
        for key in ("options", "defaults"):
            value = body.get(key)
            if value is not None and not isinstance(value, dict):
                raise invalid(f"'{key}' must be an object", f"/{key}")
            batch_shared[key] = value
        items: list[tuple[str | None, Any]] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                items.append((None, item))  # fails on its own as an invalid request
                continue
            item_id = item.get("id")
            if item_id is not None:
                if not isinstance(item_id, str):
                    raise invalid("Item 'id' must be a string", f"/items/{index}/id")
                if item_id in seen_ids:
                    raise invalid(f"Duplicate item ID {item_id!r}", f"/items/{index}/id")
                seen_ids.add(item_id)
            item_body = {key: value for key, value in item.items() if key != "id"}
            for key, batch_value in batch_shared.items():  # item values override batch values
                merged = merge_options(batch_value, item.get(key))
                if merged is None:
                    item_body.pop(key, None)
                else:
                    item_body[key] = merged
            items.append((item_id, item_body))
        return items, model

    @staticmethod
    def batch_response(
        items: list[tuple[str | None, Any]],
        model: str,
        outcomes: Sequence[dict[str, Any] | APIError],
    ) -> dict[str, Any]:
        response_items = []
        for index, ((item_id, _), outcome) in enumerate(zip(items, outcomes, strict=True)):
            entry: dict[str, Any] = {}
            if item_id is not None:
                entry["id"] = item_id
            if isinstance(outcome, APIError):
                entry.update(outcome.with_prefix(f"/items/{index}").to_dict())
            else:
                entry["results"] = outcome["results"]
                if "usage" in outcome:
                    entry["usage"] = outcome["usage"]
            response_items.append(entry)
        return {"model": model, "items": response_items}

    # -- planning -------------------------------------------------------------------

    def _prepare(
        self, bodies: Sequence[Any], *, check_model: bool
    ) -> tuple[list[dict[str, Any] | APIError | None], list[tuple[int, Plan]]]:
        outcomes: list[dict[str, Any] | APIError | None] = [None] * len(bodies)
        plans: list[tuple[int, Plan]] = []
        for index, body in enumerate(bodies):
            try:
                plans.append((index, self.plan(self.parser.parse(body, check_model=check_model))))
            except APIError as exc:
                outcomes[index] = exc
        return outcomes, plans

    def threshold_for(self, task: TaskSpec) -> float:
        if task.threshold is not None:
            return task.threshold
        return float(self.capabilities.default_thresholds.get(task.type, 0.5))

    def entity_threshold_for(self, task: TaskSpec) -> float:
        if task.entity_threshold is not None:
            return task.entity_threshold
        return float(self.capabilities.default_thresholds.get("span", 0.5))

    def plan(self, parsed: ParsedRequest) -> Plan:
        # Check every distinct adapter before anything runs; errors point to where it was set.
        checked: set[str | None] = set()
        for task in parsed.tasks:
            if task.adapter_id in checked:
                continue
            try:
                self.resolve_adapter(task.adapter_id)
            except KeyError as exc:
                raise APIError("UNKNOWN_ADAPTER", str(exc).strip("'\""), task.adapter_path) from exc
            except ValueError as exc:
                raise invalid(str(exc), task.adapter_path) from exc
            checked.add(task.adapter_id)

        plan = Plan(parsed=parsed)
        text = parsed.rendered.text
        words = self.split_words(text)
        relation_default = float(self.capabilities.default_thresholds.get("relation", 0.5))
        for task in parsed.tasks:
            if task.type == "relation":
                labels = task.entity_labels or []
                relations: list[str] | None = [label.text for label in task.labels]
                threshold = self.entity_threshold_for(task)
                relation_threshold = self.threshold_for(task)
                prompt = _prompt_words(labels) + _prompt_words(task.labels)
            else:
                labels = task.labels
                relations = None
                threshold = self.threshold_for(task)
                relation_threshold = relation_default  # unused by span rows
                prompt = _prompt_words(labels)
            budget = self.max_words - prompt
            if budget < 1:
                raise invalid(
                    f"The labels of this task fill the model window ({prompt} of {self.max_words} words)",
                    f"{task.path}/labels",
                )
            for start, end, count in self.windows(words, len(text), budget):
                row = {
                    "text": text[start:end],
                    "labels": _model_labels(labels),
                    "relations": relations,
                    "threshold": threshold,
                    "relation_threshold": relation_threshold,
                    "flat_ner": task.flat,
                    "multi_label": task.multi_label,
                    "adapter_id": task.adapter_id,
                }
                plan.refs.append(RowRef(task=task, row=row, offset=start, words=count + prompt))
        return plan

    def windows(self, words: list[tuple[int, int]], length: int, budget: int) -> list[tuple[int, int, int]]:
        """Overlapping ``(start, end, words)`` character windows of at most ``budget`` words."""
        if len(words) <= budget:
            return [(0, length, len(words))]
        step = max(1, int(budget * (1.0 - self.window_overlap)))
        result = []
        first = 0
        while True:
            last = min(first + budget, len(words))
            result.append((words[first][0], words[last - 1][1], last - first))
            if last >= len(words):
                return result
            first += step

    @staticmethod
    def _row_key(row: Row) -> tuple:
        labels = row["labels"]
        labels_key = tuple(labels.items()) if isinstance(labels, dict) else tuple(labels)
        relations = row["relations"]
        return (
            row["text"],
            labels_key,
            None if relations is None else tuple(relations),
            row["threshold"],
            row["relation_threshold"] if relations is not None else None,
            row["flat_ner"],
            row["multi_label"],
            row["adapter_id"],
        )

    def _schedule(self, plans: list[tuple[int, Plan]]) -> list[Row]:
        """Unique rows of all plans; every ``RowRef`` gets the slot of its row."""
        slots: dict[tuple, int] = {}
        rows: list[Row] = []
        for _, plan in plans:
            for ref in plan.refs:
                key = self._row_key(ref.row)
                slot = slots.get(key)
                if slot is None:
                    slot = slots[key] = len(rows)
                    rows.append(ref.row)
                ref.slot = slot
        return rows

    @staticmethod
    def _distribute(plans: list[tuple[int, Plan]], results: list[Any]) -> None:
        for _, plan in plans:
            outputs = []
            for ref in plan.refs:
                result = results[ref.slot]
                if isinstance(result, BaseException):
                    plan.error = _execution_error(result)
                    break
                outputs.append(result)
            else:
                plan.outputs = outputs

    def _finish(
        self, outcomes: list[dict[str, Any] | APIError | None], plans: list[tuple[int, Plan]]
    ) -> list[dict[str, Any] | APIError]:
        for index, plan in plans:
            if plan.error is not None:
                outcomes[index] = plan.error
                continue
            try:
                outcomes[index] = self.assemble(plan)
            except APIError as exc:
                outcomes[index] = exc
            except Exception as exc:
                logger.exception("Failed to assemble process results")
                outcomes[index] = APIError("INTERNAL", f"Result assembly failed: {exc}")
        return outcomes  # type: ignore[return-value]

    # -- assembly ---------------------------------------------------------------------

    def assemble(self, plan: Plan) -> dict[str, Any]:
        parsed = plan.parsed
        outputs = plan.outputs or []
        results: dict[str, Any] = {}
        for task in parsed.tasks:
            pairs = [(ref, outputs[index]) for index, ref in enumerate(plan.refs) if ref.task is task]
            if task.type == "relation":
                result = self.relation_result(parsed.rendered, task, pairs)
            else:
                result = self.span_result(parsed.rendered, task, pairs)
            if parsed.options.include_label_definitions:
                result["label_definitions"] = {label.id: label.definition() for label in task.labels}
                if task.entity_labels is not None:
                    result["entity_label_definitions"] = {label.id: label.definition() for label in task.entity_labels}
            results[task.name] = result
        response: dict[str, Any] = {"model": parsed.model, "results": results}
        if parsed.options.include_usage:
            response["usage"] = {
                "rows": len(plan.refs),
                "tokens": sum(ref.words for ref in plan.refs),
                "latency_ms": round((time.perf_counter() - plan.started) * 1000.0, 1),
            }
        return response

    def span_result(
        self, rendered: RenderedInput, task: TaskSpec, pairs: list[tuple[RowRef, RowOutput]]
    ) -> dict[str, Any]:
        lookup = _label_lookup(task.labels)
        merged: dict[tuple[int, int, str], dict[str, Any]] = {}
        for ref, output in pairs:
            for entity in output.get("entities") or []:
                label_id = lookup.get(entity.get("label"))
                start, end = entity.get("start"), entity.get("end")
                if label_id is None or not isinstance(start, int) or not isinstance(end, int):
                    continue
                start, end = start + ref.offset, end + ref.offset
                location = rendered.locate(start, end)
                if location is None:  # touches a heading or crosses two fields
                    continue
                score = float(entity.get("score", 0.0))
                key = (start, end, label_id)
                if key in merged and merged[key]["score"] >= score:
                    continue
                merged[key] = {"label": label_id, **location.to_dict(), "score": score, "_start": start, "_end": end}
        spans = list(merged.values())
        if task.flat and len(pairs) > 1:  # windows overlap: resolve overlaps across them
            spans = _greedy_flat(spans, multi_label=task.multi_label)
        spans.sort(key=lambda span: (span["_start"], span["_end"], span["label"]))
        for span in spans:
            del span["_start"], span["_end"]
        return {"type": "span", "value": spans, "threshold": self.threshold_for(task), "adapter_id": task.adapter_id}

    def relation_result(
        self, rendered: RenderedInput, task: TaskSpec, pairs: list[tuple[RowRef, RowOutput]]
    ) -> dict[str, Any]:
        relation_lookup = _label_lookup(task.labels)
        entity_lookup = _label_lookup(task.entity_labels)
        merged: dict[tuple, dict[str, Any]] = {}
        for ref, output in pairs:
            for relation in output.get("relations") or []:
                relation_id = relation_lookup.get(relation.get("relation"))
                if relation_id is None:
                    continue
                ends = []
                for role in ("head", "tail"):
                    entity = relation.get(role) or {}
                    start, end = entity.get("start"), entity.get("end")
                    if not isinstance(start, int) or not isinstance(end, int):
                        break
                    start, end = start + ref.offset, end + ref.offset
                    location = rendered.locate(start, end)
                    if location is None:
                        break
                    item: dict[str, Any] = {}
                    label_id = entity_lookup.get(entity.get("type", entity.get("label")))
                    if label_id is not None:
                        item["label"] = label_id
                    item.update(location.to_dict())
                    ends.append((item, start, end))
                if len(ends) != 2:
                    continue
                score = float(relation.get("score", 0.0))
                key = (ends[0][1], ends[0][2], relation_id, ends[1][1], ends[1][2])
                if key in merged and merged[key]["score"] >= score:
                    continue
                merged[key] = {
                    "head": ends[0][0],
                    "relation": relation_id,
                    "tail": ends[1][0],
                    "score": score,
                    "_order": key,
                }
        relations = sorted(merged.values(), key=lambda item: item["_order"])
        for item in relations:
            del item["_order"]
        return {
            "type": "relation",
            "value": relations,
            "threshold": self.threshold_for(task),
            "entity_threshold": self.entity_threshold_for(task),
            "adapter_id": task.adapter_id,
        }


def _execution_error(exc: BaseException) -> APIError:
    if isinstance(exc, APIError):
        return exc
    logger.error("Process execution failed: %s: %s", type(exc).__name__, exc)
    return APIError("INTERNAL", f"Model execution failed: {type(exc).__name__}: {exc}")
