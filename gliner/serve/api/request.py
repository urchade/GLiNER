"""Validation of process requests into task specs (NEW_API.md §4, §13.4).

GLiNER serves two task types:

* ``span`` -- ``labels``, ``threshold``, ``flat`` (alias ``flat_ner``) and
  ``multi_label`` (one span may carry several labels);
* ``relation`` -- relation ``labels``, ``entity_labels`` (required: GLiNER
  extracts the relation's entities in the same pass), ``threshold`` (relation
  cutoff), ``entity_threshold``, ``flat`` / ``flat_ner`` and ``multi_label``.

Every task also takes ``adapter_id`` (PolyLoRA adapter; ``null`` is the base model).

Settings shared by many tasks can be given once in the request's ``defaults``:
``adapter_id``, ``threshold``, ``entity_threshold``, ``flat`` / ``flat_ner`` and
``multi_label``. A task's own value wins; an explicit ``null`` in a task skips the
request default and uses the deployment's. A default that does not apply to a
task type (``entity_threshold`` on a ``span`` task) is skipped for it.
``options.adapter_id`` is accepted as an alias of ``defaults.adapter_id``.

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
SETTINGS = frozenset({"adapter_id", "threshold", "flat", "flat_ner", "multi_label", "instruction"})
SPAN_FIELDS = SETTINGS | {"type", "labels"}
RELATION_FIELDS = SPAN_FIELDS | {"entity_labels", "entity_threshold"}
TASK_FIELDS = {"span": SPAN_FIELDS, "relation": RELATION_FIELDS}
DEFAULT_KEYS = SETTINGS | {"entity_threshold"}
REQUEST_KEYS = frozenset({"model", "input", "tasks", "options", "defaults"})
BATCH_KEYS = frozenset({"model", "items", "options", "defaults"})


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
    adapter_id: str | None = None
    adapter_path: str | None = None  # where the adapter was set, for error paths


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


def _adapter(value: Any, where: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise invalid("'adapter_id' must be a string or null", where)
    return value


def _check_setting(key: str, value: Any, where: str) -> Any:
    """Validate one setting value (``null`` means "use the deployment default")."""
    if value is None:
        return None
    if key in ("threshold", "entity_threshold"):
        return _threshold(value, where)
    if key in ("flat", "flat_ner", "multi_label"):
        return _bool(value, key, where)
    if key == "adapter_id":
        return _adapter(value, where)
    if key == "instruction":
        raise APIError(
            "UNSUPPORTED_FIELD",
            "GLiNER is not conditioned on instructions; describe what to find with labels",
            where,
        )
    raise invalid(f"Unknown setting '{key}'", where)


def parse_defaults(raw: Any, options: Options, path: str = "/defaults") -> dict[str, tuple[Any, str]]:
    """Request-level settings as ``{key: (value, path)}``; ``flat_ner`` is stored as ``flat``."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise invalid("'defaults' must be an object", path)
    defaults: dict[str, tuple[Any, str]] = {}
    for key, raw_value in raw.items():
        if key not in DEFAULT_KEYS:
            raise invalid(f"Unknown default '{key}'", f"{path}/{key}")
        where = f"{path}/{key}"
        value = _check_setting(key, raw_value, where)
        name = "flat" if key == "flat_ner" else key
        if name in defaults and defaults[name][0] != value:
            raise invalid("'flat' and 'flat_ner' disagree; give only one of them", f"{path}/flat_ner")
        defaults[name] = (value, where)
    if options.adapter_id is not None:
        if "adapter_id" in defaults and defaults["adapter_id"][0] != options.adapter_id:
            raise invalid("'options.adapter_id' and 'defaults.adapter_id' disagree", "/options/adapter_id")
        defaults.setdefault("adapter_id", (options.adapter_id, "/options/adapter_id"))
    return defaults


def merge_options(batch: Any, item: Any) -> Any:
    """Item-level options (or defaults) override batch-level ones."""
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
        defaults = parse_defaults(body.get("defaults"), options)
        tasks_raw = body.get("tasks")
        if not isinstance(tasks_raw, dict) or not tasks_raw:
            raise invalid("'tasks' must be a non-empty object", "/tasks")
        max_tasks = self.capabilities.limit("max_tasks")
        if max_tasks and len(tasks_raw) > max_tasks:
            raise invalid(f"At most {max_tasks} tasks are allowed per request", "/tasks")
        tasks = [self.parse_task(name, raw, pointer("tasks", name), defaults) for name, raw in tasks_raw.items()]
        rendered = self.render(body["input"])
        return ParsedRequest(model=model, rendered=rendered, tasks=tasks, options=options)

    # -- tasks ----------------------------------------------------------------

    def parse_task(
        self, name: str, raw: Any, path: str, defaults: dict[str, tuple[Any, str]] | None = None
    ) -> TaskSpec:
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
        defaults = defaults or {}

        def setting(key: str) -> tuple[Any, str | None]:
            """The task's value, else the request default; ``(None, None)`` when neither is set."""
            keys = ("flat", "flat_ner") if key == "flat" else (key,)
            given = {k: _check_setting(k, raw[k], f"{path}/{k}") for k in keys if k in raw}
            if len(set(given.values())) > 1:
                raise invalid("'flat' and 'flat_ner' disagree; give only one of them", f"{path}/flat_ner")
            if given:
                key_used = next(iter(given))
                return given[key_used], f"{path}/{key_used}"
            return defaults.get(key, (None, None))

        setting("instruction")  # rejected when set
        task = TaskSpec(name=name, type=task_type, path=path, labels=[])
        task.threshold = setting("threshold")[0]
        flat = setting("flat")[0]
        task.flat = True if flat is None else flat
        task.multi_label = bool(setting("multi_label")[0])
        task.adapter_id, adapter_path = setting("adapter_id")
        task.adapter_path = adapter_path or f"{path}/adapter_id"
        task.labels = self.labels(raw.get("labels"), f"{path}/labels", "labels")
        if task_type == "relation":
            if raw.get("entity_labels") is None:
                raise invalid(
                    "GLiNER 'relation' tasks need 'entity_labels': the entity types the relations connect",
                    f"{path}/entity_labels",
                )
            task.entity_labels = self.labels(raw["entity_labels"], f"{path}/entity_labels", "entity_labels")
            task.entity_threshold = setting("entity_threshold")[0]
        return task

    def labels(self, spec: Any, where: str, name: str) -> list[Label]:
        labels = normalize_labels(spec, where, template=self.capabilities.label_template)
        if not labels:
            # Open extraction (``labels: null``) needs instructions, which GLiNER does not take.
            raise invalid(f"'{name}' must name at least one label", where)
        limit = self.capabilities.limit("max_labels_per_task")
        if limit and len(labels) > limit:
            raise invalid(f"At most {limit} labels are allowed per task", where)
        return labels
