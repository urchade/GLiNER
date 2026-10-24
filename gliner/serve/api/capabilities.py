"""What the served GLiNER checkpoint can run (NEW_API.md §13.2)."""

from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

from .labels import DEFAULT_LABEL_TEMPLATE

# Every task type of the unified API. GLiNER serves only ``span`` and, on
# relation-extraction checkpoints, ``relation``; the rest are rejected with
# ``UNSUPPORTED_TASK_TYPE``.
ALL_TASK_TYPES = (
    "classify",
    "boolean",
    "score",
    "span",
    "relation",
    "json",
    "embedding",
    "rank",
    "box",
    "mask",
    "count",
    "time_span",
)
RESERVED_TASK_TYPES = ("text", "link")


@dataclass
class Capabilities:
    model: str
    aliases: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=lambda: ["span"])
    label_template: str = DEFAULT_LABEL_TEMPLATE
    default_thresholds: dict[str, float] = field(default_factory=dict)
    limits: dict[str, int] = field(default_factory=dict)
    adapters_enabled: bool = False

    def has_task(self, task_type: str) -> bool:
        return task_type in self.task_types

    def model_ids(self) -> list[str]:
        return [self.model, *self.aliases]

    def limit(self, name: str) -> int:
        """A limit, with ``0`` for "unlimited"."""
        value = self.limits.get(name) or 0
        return value if value > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "aliases": list(self.aliases),
            "modalities": ["text"],
            "task_types": list(self.task_types),
            "json_field_types": [],
            "json_text_formats": [],
            "label_template": self.label_template,
            "default_thresholds": dict(self.default_thresholds),
            # Non-positive limits mean "unlimited" and are reported as null.
            "limits": {key: (value if value and value > 0 else None) for key, value in self.limits.items()},
            "adapters": {"enabled": self.adapters_enabled},
        }


def capabilities_from_config(config: Any, *, supports_relations: bool) -> Capabilities:
    """Capabilities of a GLiNER deployment from its serve config."""
    model_id = getattr(config, "model_id", None) or config.model
    aliases = [alias for alias in [config.model, *(getattr(config, "model_aliases", None) or [])] if alias != model_id]
    task_types = ["span", "relation"] if supports_relations else ["span"]
    thresholds = {"span": float(config.default_threshold)}
    if supports_relations:
        thresholds["relation"] = float(config.default_relation_threshold)
    return Capabilities(
        model=model_id,
        aliases=list(dict.fromkeys(aliases)),
        task_types=task_types,
        label_template=getattr(config, "label_template", DEFAULT_LABEL_TEMPLATE),
        default_thresholds=thresholds,
        limits={
            "max_input_tokens": int(config.max_model_len),
            "max_input_chars": int(getattr(config, "max_input_chars", 0)),
            "max_input_bytes": int(getattr(config, "max_request_bytes", 0)),
            "max_tasks": int(getattr(config, "max_tasks", 0)),
            "max_labels_per_task": int(config.max_labels),
            "max_batch_items": int(getattr(config, "max_batch_items", 0)),
        },
        adapters_enabled=bool(config.enable_polylora),
    )
