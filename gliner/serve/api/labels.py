"""Label specs: accepted forms, definitions and the text the model sees (NEW_API.md §5)."""

from __future__ import annotations

import string
from typing import Any
from dataclasses import dataclass

from .errors import APIError, invalid, escape_pointer_token

DEFINITION_KEYS = frozenset({"title", "description", "metadata"})
UNSUPPORTED_DEFINITION_KEYS = frozenset({"examples", "aliases"})
DEFAULT_LABEL_TEMPLATE = "{title}"


@dataclass
class Label:
    """One normalized label."""

    id: str
    text: str  # what the model sees
    title: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None

    def definition(self) -> dict[str, Any]:
        """The resolved definition returned with ``include_label_definitions``."""
        result: dict[str, Any] = {}
        if self.title is not None:
            result["title"] = self.title
        if self.description is not None:
            result["description"] = self.description
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result


def validate_template(template: str) -> str:
    """Check a deployment label template such as ``"{title}: {description}"``."""
    allowed = {"id", "title", "description"}
    try:
        names = {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
    except ValueError as exc:
        raise ValueError(f"Invalid label_template {template!r}: {exc}") from exc
    unknown = names - allowed
    if unknown:
        raise ValueError(f"label_template may only use {sorted(allowed)}, got {sorted(unknown)}")
    if "title" not in names and "id" not in names:
        raise ValueError("label_template must include {title} or {id}")
    return template


def render_label(label_id: str, title: str | None, description: str | None, template: str) -> str:
    """Text shown to the model for one label."""
    shown_title = title if title is not None else label_id
    if description is None and "{description}" in template:
        template = DEFAULT_LABEL_TEMPLATE
    text = template.format(id=label_id, title=shown_title, description=description or "")
    return " ".join(text.split())


def normalize_labels(spec: Any, path: str, *, template: str = DEFAULT_LABEL_TEMPLATE) -> list[Label] | None:
    """Normalize the accepted label forms into ``Label`` objects.

    Forms: ``None``, ``"id"``, ``["id", ...]``, ``{"id": "description"}`` and
    ``{"id": {"title", "description", "metadata"}}``.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        items: list[tuple[Any, Any, str]] = [(spec, None, path)]
    elif isinstance(spec, list):
        items = []
        for index, label_id in enumerate(spec):
            if not isinstance(label_id, str):
                raise invalid("Label lists must contain strings", f"{path}/{index}")
            items.append((label_id, None, f"{path}/{index}"))
    elif isinstance(spec, dict):
        items = [(label_id, value, f"{path}/{escape_pointer_token(label_id)}") for label_id, value in spec.items()]
    else:
        raise invalid("labels must be null, a string, a list of strings, or an object", path)

    labels: list[Label] = []
    seen_ids: set[str] = set()
    rendered: dict[str, str] = {}
    for label_id, value, where in items:
        if not isinstance(label_id, str) or not label_id.strip():
            raise invalid("Label IDs must be non-empty strings", where)
        if label_id in seen_ids:
            raise invalid(f"Duplicate label ID {label_id!r}", where)
        seen_ids.add(label_id)
        title = description = None
        metadata = None
        if isinstance(value, str):
            description = value
        elif isinstance(value, dict):
            for key in value:
                if key in DEFINITION_KEYS:
                    continue
                if key in UNSUPPORTED_DEFINITION_KEYS:
                    raise APIError(
                        "UNSUPPORTED_FIELD", f"Label property '{key}' is not used by this model", f"{where}/{key}"
                    )
                raise invalid(f"Unknown label property '{key}'", f"{where}/{key}")
            title = value.get("title")
            description = value.get("description")
            metadata = value.get("metadata")
            for key, prop in (("title", title), ("description", description)):
                if prop is not None and not isinstance(prop, str):
                    raise invalid(f"'{key}' must be a string", f"{where}/{key}")
            if title is not None and not title.strip():
                raise invalid("'title' must not be empty", f"{where}/title")
            if metadata is not None and not isinstance(metadata, dict):
                raise invalid("'metadata' must be an object", f"{where}/metadata")
        elif value is not None:
            raise invalid("A label definition must be a description string or an object", where)
        text = render_label(label_id, title, description, template)
        if not text:
            raise invalid(f"Label {label_id!r} renders to empty text", where)
        if text in rendered:
            raise invalid(f"Labels {rendered[text]!r} and {label_id!r} render to the same text {text!r}", where)
        rendered[text] = label_id
        labels.append(Label(id=label_id, text=text, title=title, description=description, metadata=metadata))
    return labels
