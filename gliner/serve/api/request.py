"""Validation of analyze requests into task specs (NEW_API.md §4, §13.4).

GLiNER serves two task types:

* ``span`` -- ``labels``, ``threshold``, ``flat`` (alias ``flat_ner``) and
  ``multi_label`` (one span may carry several labels);
* ``relation`` -- relation ``labels``, ``entity_labels`` (required: GLiNER
  extracts the relation's entities in the same pass), ``threshold`` (relation
  cutoff), ``entity_threshold``, ``flat`` / ``flat_ner`` and ``multi_label``.

GLiNER is not conditioned on instructions, so a non-null ``instruction`` is
rejected with ``UNSUPPORTED_FIELD`` instead of being ignored.
"""

from __future__ import annotations

import math
from typing import Any
from datetime import date, datetime
from dataclasses import dataclass

from .errors import APIError, invalid, pointer
from .labels import Label, normalize_labels
from .render import RenderedInput, render_input
from .capabilities import ALL_TASK_TYPES, RESERVED_TASK_TYPES, Capabilities

OPTION_KEYS = frozenset(
    {
        "include_label_definitions",
        "include_usage",
        "include_evidence",
        "include_layout_text",
        "reference_date",
        "adapter_id",
    }
)
# ``include_evidence`` / ``include_layout_text`` / ``reference_date`` belong to
# task and input types GLiNER does not serve; they are validated and have no effect.
BOOLEAN_OPTIONS = ("include_label_definitions", "include_usage", "include_evidence", "include_layout_text")
SPAN_FIELDS = frozenset({"type", "instruction", "labels", "threshold", "flat", "flat_ner", "multi_label"})
RELATION_FIELDS = SPAN_FIELDS | {"entity_labels", "entity_threshold"}
TASK_FIELDS = {"span": SPAN_FIELDS, "relation": RELATION_FIELDS}
REQUEST_KEYS = frozenset({"model", "input", "tasks", "options"})
BATCH_KEYS = frozenset({"model", "items", "options"})


@dataclass
class TaskSpec:
    name: str
    type: str
    path: str
    labels: list[Label]
    threshold: float | None = None
    flat: bool = True
    multi_label: bool = False
    entity_labels: list[Label] | None = None  # relation only
    entity_threshold: float | None = None  # relation only


@dataclass
class Options:
    include_label_definitions: bool = False
    include_usage: bool = False
    include_evidence: bool = False
    include_layout_text: bool = False
    reference_date: str | None = None
    adapter_id: str | None = None


@dataclass
class ParsedRequest:
    model: str
    rendered: RenderedInput
    tasks: list[TaskSpec]
    options: Options


def _threshold(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise invalid("A threshold must be a number", where)
    if not 0.0 <= float(value) <= 1.0:
        raise invalid("A threshold must be between 0 and 1", where)
    return float(value)


def _bool(value: Any, name: str, where: str) -> bool:
    if not isinstance(value, bool):
        raise invalid(f"'{name}' must be a boolean", where)
    return value


def parse_options(raw: Any, path: str = "/options") -> Options:
    if raw is None:
        return Options()
    if not isinstance(raw, dict):
        raise invalid("'options' must be an object", path)
    for key in raw:
        if key not in OPTION_KEYS:
            raise invalid(f"Unknown option '{key}'", f"{path}/{key}")
    options = Options()
    for key in BOOLEAN_OPTIONS:
        if key in raw:
            setattr(options, key, _bool(raw[key], key, f"{path}/{key}"))
    if raw.get("reference_date") is not None:
        value = raw["reference_date"]
        if not isinstance(value, str):
            raise invalid("'reference_date' must be an ISO 8601 string", f"{path}/reference_date")
        try:
            (datetime if "T" in value else date).fromisoformat(value)
        except ValueError as exc:
            raise invalid(f"'reference_date' is not ISO 8601: {exc}", f"{path}/reference_date") from exc
        options.reference_date = value
    if raw.get("adapter_id") is not None:
        if not isinstance(raw["adapter_id"], str):
            raise invalid("'adapter_id' must be a string", f"{path}/adapter_id")
        options.adapter_id = raw["adapter_id"]
    return options


def merge_options(batch: Any, item: Any) -> Any:
    """Item-level options override batch-level ones."""
    if batch is None:
        return item
    if item is None:
        return batch
    if not isinstance(batch, dict) or not isinstance(item, dict):
        return item
    return {**batch, **item}


class RequestParser:
    """Validate requests against the contract and the model's capabilities."""

    def __init__(self, capabilities: Capabilities) -> None:
        self.capabilities = capabilities

    def check_model(self, model: Any, path: str = "/model") -> str:
        if not isinstance(model, str) or not model:
            raise invalid("'model' must be a non-empty string", path)
        if model not in self.capabilities.model_ids():
            raise APIError(
                "MODEL_NOT_FOUND",
                f"Model {model!r} is not served here",
                path,
                {"available": self.capabilities.model_ids()},
            )
        return self.capabilities.model

    def render(self, value: Any) -> RenderedInput:
        rendered = render_input(value)
        limit = self.capabilities.limit("max_input_chars")
        if limit and len(rendered.text) > limit:
            raise APIError(
                "INPUT_TOO_LONG",
                f"Rendered input has {len(rendered.text)} characters; the limit is {limit}",
                "/input",
            )
        return rendered

    def parse(self, body: Any, *, check_model: bool = True) -> ParsedRequest:
        if not isinstance(body, dict):
            raise invalid("The request body must be a JSON object", "")
        for key in body:
            if key not in REQUEST_KEYS:
                raise invalid(f"Unknown request field '{key}'", pointer(key))
        model = self.check_model(body.get("model")) if check_model else self.capabilities.model
        if "input" not in body:
            raise invalid("'input' is required", "/input")
        options = parse_options(body.get("options"))
        tasks_raw = body.get("tasks")
        if not isinstance(tasks_raw, dict) or not tasks_raw:
            raise invalid("'tasks' must be a non-empty object", "/tasks")
        max_tasks = self.capabilities.limit("max_tasks")
        if max_tasks and len(tasks_raw) > max_tasks:
            raise invalid(f"At most {max_tasks} tasks are allowed per request", "/tasks")
        tasks = [self.parse_task(name, raw, pointer("tasks", name)) for name, raw in tasks_raw.items()]
        rendered = self.render(body["input"])
        return ParsedRequest(model=model, rendered=rendered, tasks=tasks, options=options)

    # -- tasks ----------------------------------------------------------------

    def parse_task(self, name: str, raw: Any, path: str) -> TaskSpec:
        if not name:
            raise invalid("Task names must be non-empty", path)
        if not isinstance(raw, dict):
            raise invalid("A task must be an object", path)
        task_type = raw.get("type")
        if task_type is None:
            raise invalid("Every task needs a 'type'", f"{path}/type")
        if task_type in RESERVED_TASK_TYPES or (task_type in ALL_TASK_TYPES and task_type not in TASK_FIELDS):
            raise APIError(
                "UNSUPPORTED_TASK_TYPE",
                f"GLiNER does not serve '{task_type}' tasks",
                f"{path}/type",
                {"model": self.capabilities.model, "supported": list(self.capabilities.task_types)},
            )
        if task_type not in ALL_TASK_TYPES:
            raise invalid(
                f"Unknown task type {task_type!r}; expected one of {', '.join(ALL_TASK_TYPES)}", f"{path}/type"
            )
        if not self.capabilities.has_task(task_type):
            raise APIError(
                "UNSUPPORTED_TASK_TYPE",
                f"The model has no head for '{task_type}' tasks",
                f"{path}/type",
                {"model": self.capabilities.model, "supported": list(self.capabilities.task_types)},
            )
        for key in raw:
            if key not in TASK_FIELDS[task_type]:
                raise invalid(f"Field '{key}' is not valid for '{task_type}' tasks", f"{path}/{key}")
        if raw.get("instruction") is not None:
            raise APIError(
                "UNSUPPORTED_FIELD",
                "GLiNER is not conditioned on instructions; describe what to find with labels",
                f"{path}/instruction",
            )

        task = TaskSpec(name=name, type=task_type, path=path, labels=[])
        if raw.get("threshold") is not None:
            task.threshold = _threshold(raw["threshold"], f"{path}/threshold")
        task.flat = self.flat(raw, path)
        if "multi_label" in raw:
            task.multi_label = _bool(raw["multi_label"], "multi_label", f"{path}/multi_label")
        task.labels = self.labels(raw.get("labels"), f"{path}/labels", "labels")
        if task_type == "relation":
            if raw.get("entity_labels") is None:
                raise invalid(
                    "GLiNER 'relation' tasks need 'entity_labels': the entity types the relations connect",
                    f"{path}/entity_labels",
                )
            task.entity_labels = self.labels(raw["entity_labels"], f"{path}/entity_labels", "entity_labels")
            if raw.get("entity_threshold") is not None:
                task.entity_threshold = _threshold(raw["entity_threshold"], f"{path}/entity_threshold")
        return task

    @staticmethod
    def flat(raw: dict[str, Any], path: str) -> bool:
        values = {key: _bool(raw[key], key, f"{path}/{key}") for key in ("flat", "flat_ner") if key in raw}
        if len(set(values.values())) > 1:
            raise invalid("'flat' and 'flat_ner' disagree; give only one of them", f"{path}/flat_ner")
        return next(iter(values.values()), True)

    def labels(self, spec: Any, where: str, name: str) -> list[Label]:
        labels = normalize_labels(spec, where, template=self.capabilities.label_template)
        if not labels:
            # Open extraction (``labels: null``) needs instructions, which GLiNER does not take.
            raise invalid(f"'{name}' must name at least one label", where)
        limit = self.capabilities.limit("max_labels_per_task")
        if limit and len(labels) > limit:
            raise invalid(f"At most {limit} labels are allowed per task", where)
        return labels
