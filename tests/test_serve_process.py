"""Process API (``/v1/process``) for GLiNER serving: validation, planning, batching, assembly.

The engine is tested with a fake row runner that "recognizes" a fixed lexicon,
so no model, GPU or Ray is needed.
"""

import re
import asyncio
from unittest.mock import Mock

import pytest

from gliner.serve.api import APIError, ProcessEngine, render_input
from gliner.serve.server import GLiNERServer
from gliner.serve.api.labels import normalize_labels
from gliner.serve.api.capabilities import Capabilities

MODEL = "test/gliner"
LEXICON = {
    "Alice": "person",
    "Bob": "person",
    "Acme": "organization",
    "London": "location",
    "Paris": "location",
}


class FakeModel:
    """Finds lexicon words whose label (or label text) the row asks for."""

    def __init__(self):
        self.calls = []

    def entities(self, row):
        labels = row["labels"]
        wanted = labels if isinstance(labels, list) else list(labels)
        texts = {} if isinstance(labels, list) else {text: label_id for label_id, text in labels.items()}
        found = []
        for match in re.finditer(r"\w+", row["text"]):
            kind = LEXICON.get(match.group())
            label = kind if kind in wanted else texts.get(kind)
            if label is not None:
                found.append(
                    {"start": match.start(), "end": match.end(), "text": match.group(), "label": label, "score": 0.9}
                )
        return found

    def run_row(self, row):
        output = {"entities": self.entities(row)}
        if row["relations"]:
            persons = [e for e in output["entities"] if e["label"] == "person"]
            orgs = [e for e in output["entities"] if e["label"] == "organization"]
            output["relations"] = [
                {
                    "head": {**head, "type": head["label"]},
                    "tail": {**tail, "type": tail["label"]},
                    "relation": row["relations"][0],
                    "score": 0.8,
                }
                for head in persons
                for tail in orgs
            ]
        return output

    def run_rows(self, rows):
        self.calls.append([dict(row) for row in rows])
        return [self.run_row(row) for row in rows]


def make_engine(*, relations=True, max_words=512, resolve_adapter=None, **limits):
    capabilities = Capabilities(
        model=MODEL,
        aliases=["alias/gliner"],
        task_types=["span", "relation"] if relations else ["span"],
        default_thresholds={"span": 0.5, "relation": 0.6} if relations else {"span": 0.5},
        limits={"max_tasks": 8, "max_labels_per_task": 5, "max_batch_items": 4, "max_input_chars": 10_000, **limits},
    )
    return ProcessEngine(capabilities, max_words=max_words, resolve_adapter=resolve_adapter)


def process(engine, input, tasks, **extra):
    model = FakeModel()
    response = engine.process({"model": MODEL, "input": input, "tasks": tasks, **extra}, model.run_rows)
    return response, model


def error_of(engine, body, model=None):
    with pytest.raises(APIError) as info:
        engine.process(body, (model or FakeModel()).run_rows)
    return info.value


# -- spans ---------------------------------------------------------------------


def test_span_task_on_string_input_reports_locations_and_threshold():
    response, _ = process(make_engine(), "Alice works at Acme.", {"ents": {"type": "span", "labels": ["person"]}})

    assert response == {
        "model": MODEL,
        "results": {
            "ents": {
                "type": "span",
                "value": [{"label": "person", "text": "Alice", "start": 0, "end": 5, "source": "", "score": 0.9}],
                "threshold": 0.5,
                "adapter_id": None,
            }
        },
    }


def test_every_task_is_one_row_and_all_rows_run_in_one_batch():
    tasks = {
        "people": {"type": "span", "labels": ["person"], "threshold": 0.3},
        "orgs": {"type": "span", "labels": "organization", "flat_ner": False, "multi_label": True},
        "places": {"type": "span", "labels": {"location": {"title": "place"}}},
    }
    response, model = process(make_engine(), "Alice met Bob at Acme in Paris.", tasks)

    assert len(model.calls) == 1  # one runner call: the rows share the batch
    rows = model.calls[0]
    assert [row["labels"] for row in rows] == [["person"], ["organization"], {"location": "place"}]
    assert [row["threshold"] for row in rows] == [0.3, 0.5, 0.5]
    assert [row["flat_ner"] for row in rows] == [True, False, True]
    assert [row["multi_label"] for row in rows] == [False, True, False]
    assert all(row["relations"] is None for row in rows)
    results = response["results"]
    assert [span["text"] for span in results["people"]["value"]] == ["Alice", "Bob"]
    assert results["people"]["threshold"] == 0.3
    assert [span["text"] for span in results["orgs"]["value"]] == ["Acme"]
    assert results["places"]["value"][0]["label"] == "location"  # the id, not the title shown to the model


def test_identical_tasks_share_one_row():
    tasks = {
        "a": {"type": "span", "labels": ["person"]},
        "b": {"type": "span", "labels": ["person"], "flat": True, "threshold": 0.5},
    }
    response, model = process(make_engine(), "Alice", tasks)

    assert len(model.calls[0]) == 1
    assert response["results"]["a"]["value"] == response["results"]["b"]["value"]


def test_structured_input_reports_source_pointers_and_leaf_offsets():
    data = {"ticket": "Bob emailed Acme.", "customer": {"name": "Alice"}, "cities": ["London", "Paris"]}
    response, model = process(make_engine(), data, {"ents": {"type": "span", "labels": ["person", "location"]}})

    assert model.calls[0][0]["text"] == render_input(data).text
    assert response["results"]["ents"]["value"] == [
        {"label": "person", "text": "Bob", "start": 0, "end": 3, "source": "/ticket", "score": 0.9},
        {"label": "person", "text": "Alice", "start": 0, "end": 5, "source": "/customer/name", "score": 0.9},
        {"label": "location", "text": "London", "start": 0, "end": 6, "source": "/cities/0", "score": 0.9},
        {"label": "location", "text": "Paris", "start": 0, "end": 5, "source": "/cities/1", "score": 0.9},
    ]


def test_spans_on_headings_are_dropped():
    response, _ = process(make_engine(), {"Alice": "Bob"}, {"ents": {"type": "span", "labels": ["person"]}})

    assert [span["text"] for span in response["results"]["ents"]["value"]] == ["Bob"]


def test_render_follows_markdown_rules():
    rendered = render_input({"a": "x", "b": {"c": 1}, "d": [{"e": True}], "f": []})

    assert rendered.text == "# a\nx\n\n# b\n## c\n1\n\n# d\n## 1\n### e\ntrue\n\n# f"
    assert [leaf.pointer for leaf in rendered.leaves] == ["/a", "/b/c", "/d/0/e"]


# -- relations -------------------------------------------------------------------


def test_relation_task_runs_entities_and_relations_in_one_row():
    tasks = {
        "rels": {
            "type": "relation",
            "labels": {"works_at": {"title": "works at", "metadata": {"uri": "urn:x"}}},
            "entity_labels": ["person", "organization"],
            "threshold": 0.7,
            "entity_threshold": 0.2,
        }
    }
    response, model = process(make_engine(), "Alice works at Acme.", tasks, options={"include_label_definitions": True})

    (row,) = model.calls[0]
    assert row["labels"] == ["person", "organization"]
    assert row["relations"] == ["works at"]  # relation text the model sees
    assert (row["threshold"], row["relation_threshold"]) == (0.2, 0.7)
    result = response["results"]["rels"]
    assert result["value"] == [
        {
            "head": {"label": "person", "text": "Alice", "start": 0, "end": 5, "source": ""},
            "relation": "works_at",
            "tail": {"label": "organization", "text": "Acme", "start": 15, "end": 19, "source": ""},
            "score": 0.8,
        }
    ]
    assert (result["threshold"], result["entity_threshold"]) == (0.7, 0.2)
    assert result["label_definitions"] == {"works_at": {"title": "works at", "metadata": {"uri": "urn:x"}}}
    assert result["entity_label_definitions"] == {"person": {}, "organization": {}}


def test_relation_defaults_come_from_capabilities():
    tasks = {"rels": {"type": "relation", "labels": "works_at", "entity_labels": ["person", "organization"]}}
    response, model = process(make_engine(), "Alice works at Acme.", tasks)

    assert (model.calls[0][0]["threshold"], model.calls[0][0]["relation_threshold"]) == (0.5, 0.6)
    assert response["results"]["rels"]["threshold"] == 0.6


def test_span_and_relation_tasks_on_one_input_batch_together():
    tasks = {
        "ents": {"type": "span", "labels": ["person", "organization"]},
        "rels": {"type": "relation", "labels": ["works_at"], "entity_labels": ["person", "organization"]},
    }
    response, model = process(make_engine(), "Alice works at Acme.", tasks)

    assert len(model.calls) == 1
    assert [row["relations"] for row in model.calls[0]] == [None, ["works_at"]]
    assert len(response["results"]["ents"]["value"]) == 2
    assert len(response["results"]["rels"]["value"]) == 1


# -- validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task", "code", "path"),
    [
        ({"type": "classify", "labels": ["a"]}, "UNSUPPORTED_TASK_TYPE", "/tasks/t/type"),
        ({"type": "text"}, "UNSUPPORTED_TASK_TYPE", "/tasks/t/type"),
        ({"type": "nope"}, "INVALID_REQUEST", "/tasks/t/type"),
        ({"labels": ["a"]}, "INVALID_REQUEST", "/tasks/t/type"),
        ({"type": "span", "labels": ["a"], "instruction": "find"}, "UNSUPPORTED_FIELD", "/tasks/t/instruction"),
        ({"type": "span", "labels": ["a"], "entity_labels": ["b"]}, "INVALID_REQUEST", "/tasks/t/entity_labels"),
        ({"type": "span", "labels": None}, "INVALID_REQUEST", "/tasks/t/labels"),
        ({"type": "span", "labels": []}, "INVALID_REQUEST", "/tasks/t/labels"),
        ({"type": "span", "labels": ["a", "a"]}, "INVALID_REQUEST", "/tasks/t/labels/1"),
        ({"type": "span", "labels": list("abcdef")}, "INVALID_REQUEST", "/tasks/t/labels"),
        ({"type": "span", "labels": ["a"], "threshold": 2}, "INVALID_REQUEST", "/tasks/t/threshold"),
        ({"type": "span", "labels": ["a"], "flat": True, "flat_ner": False}, "INVALID_REQUEST", "/tasks/t/flat_ner"),
        ({"type": "span", "labels": ["a"], "flat": "yes"}, "INVALID_REQUEST", "/tasks/t/flat"),
        ({"type": "span", "labels": {"a": {"examples": ["x"]}}}, "UNSUPPORTED_FIELD", "/tasks/t/labels/a/examples"),
        ({"type": "span", "labels": {"a": {"title": "b"}, "b": None}}, "INVALID_REQUEST", "/tasks/t/labels/b"),
        ({"type": "relation", "labels": ["r"]}, "INVALID_REQUEST", "/tasks/t/entity_labels"),
        ({"type": "relation", "entity_labels": ["a"]}, "INVALID_REQUEST", "/tasks/t/labels"),
    ],
)
def test_invalid_tasks_are_rejected_with_a_path(task, code, path):
    error = error_of(make_engine(), {"model": MODEL, "input": "x", "tasks": {"t": task}})

    assert (error.code, error.path) == (code, path)


def test_relation_task_needs_a_relation_model():
    task = {"type": "relation", "labels": ["r"], "entity_labels": ["a"]}
    error = error_of(make_engine(relations=False), {"model": MODEL, "input": "x", "tasks": {"t": task}})

    assert (error.code, error.status, error.details["supported"]) == ("UNSUPPORTED_TASK_TYPE", 422, ["span"])


@pytest.mark.parametrize(
    ("body", "code", "path"),
    [
        ({"input": "x", "tasks": {"t": {"type": "span", "labels": "a"}}}, "INVALID_REQUEST", "/model"),
        (
            {"model": "other", "input": "x", "tasks": {"t": {"type": "span", "labels": "a"}}},
            "MODEL_NOT_FOUND",
            "/model",
        ),
        ({"model": MODEL, "tasks": {"t": {"type": "span", "labels": "a"}}}, "INVALID_REQUEST", "/input"),
        ({"model": MODEL, "input": "x", "tasks": {}}, "INVALID_REQUEST", "/tasks"),
        ({"model": MODEL, "input": "x", "tasks": {"t": {"type": "span", "labels": "a"}}, "extra": 1}, None, "/extra"),
        (
            {"model": MODEL, "input": "x", "tasks": {"t": {"type": "span", "labels": "a"}}, "options": {"bad": 1}},
            "INVALID_REQUEST",
            "/options/bad",
        ),
        (
            {
                "model": MODEL,
                "input": {"p": {"$type": "image", "data": "x"}},
                "tasks": {"t": {"type": "span", "labels": "a"}},
            },
            "UNSUPPORTED_MODALITY",
            "/input/p",
        ),
        (
            {"model": MODEL, "input": {"p": {"$type": "video"}}, "tasks": {"t": {"type": "span", "labels": "a"}}},
            "INVALID_REQUEST",
            "/input/p/$type",
        ),
        (
            {"model": MODEL, "input": "x" * 10_001, "tasks": {"t": {"type": "span", "labels": "a"}}},
            "INPUT_TOO_LONG",
            "/input",
        ),
    ],
)
def test_invalid_requests_are_rejected_with_a_path(body, code, path):
    error = error_of(make_engine(), body)

    assert error.code == (code or "INVALID_REQUEST")
    assert error.path == path


def test_model_alias_is_accepted_and_canonical_id_returned():
    model = FakeModel()
    response = make_engine().process(
        {"model": "alias/gliner", "input": "Alice", "tasks": {"t": {"type": "span", "labels": "person"}}},
        model.run_rows,
    )

    assert response["model"] == MODEL


def test_label_text_follows_template():
    labels = normalize_labels(
        {"person": {"title": "Person", "description": "a human"}, "org": "a company", "loc": None},
        "/labels",
        template="{title}: {description}",
    )

    assert [label.text for label in labels] == ["Person: a human", "org: a company", "loc"]


# -- adapters ----------------------------------------------------------------------


def test_adapter_id_is_passed_to_every_row():
    seen = []
    engine = make_engine(resolve_adapter=seen.append)
    tasks = {"a": {"type": "span", "labels": "person"}, "b": {"type": "span", "labels": "location"}}
    _, model = process(engine, "Alice in Paris", tasks, options={"adapter_id": "legal"})

    assert seen == ["legal"]
    assert [row["adapter_id"] for row in model.calls[0]] == ["legal", "legal"]


def test_unknown_adapter_is_rejected_before_running():
    def resolve(adapter_id):
        if adapter_id is not None:
            raise KeyError(f"Unknown LoRA adapter id: {adapter_id}")

    model = FakeModel()
    body = {
        "model": MODEL,
        "input": "x",
        "tasks": {"t": {"type": "span", "labels": "a"}},
        "options": {"adapter_id": "z"},
    }
    error = error_of(make_engine(resolve_adapter=resolve), body, model)

    assert (error.code, error.status, error.path) == ("UNKNOWN_ADAPTER", 404, "/options/adapter_id")
    assert model.calls == []


# -- long inputs -------------------------------------------------------------------


def test_long_input_runs_over_overlapping_windows_and_merges_spans():
    text = " ".join(["Alice"] + ["filler"] * 20 + ["Paris"] + ["filler"] * 20 + ["Bob"])
    # 2 prompt words (one label + its marker) leave 10 text words per window.
    engine = make_engine(max_words=12)
    response, model = process(engine, text, {"t": {"type": "span", "labels": ["person", "location"]}})

    rows = model.calls[0]
    assert len(rows) > 1
    assert all(len(row["text"].split()) <= 9 for row in rows)  # 3 prompt words: 2 labels and their markers... per row
    spans = response["results"]["t"]["value"]
    assert [(span["text"], span["start"]) for span in spans] == [
        ("Alice", 0),
        ("Paris", text.index("Paris")),
        ("Bob", text.index("Bob")),
    ]


def test_labels_that_fill_the_window_are_rejected():
    error = error_of(
        make_engine(max_words=4),
        {"model": MODEL, "input": "x", "tasks": {"t": {"type": "span", "labels": ["a b", "c"]}}},
    )

    assert (error.code, error.path) == ("INVALID_REQUEST", "/tasks/t/labels")


# -- batch ---------------------------------------------------------------------------


def test_batch_keeps_order_ids_and_per_item_errors_and_dedupes_rows():
    task = {"t": {"type": "span", "labels": ["person"]}}
    body = {
        "model": MODEL,
        "items": [
            {"id": "a", "input": "Alice", "tasks": task},
            {"id": "b", "input": "x", "tasks": {"t": {"type": "classify", "labels": ["a"]}}},
            {"id": "c", "input": "Alice", "tasks": task},
            {"input": "Bob", "tasks": task, "options": {"include_usage": True}},
        ],
        "options": {"adapter_id": None},
    }
    model = FakeModel()
    response = make_engine().process_batch(body, model.run_rows)

    assert len(model.calls) == 1 and len(model.calls[0]) == 2  # "Alice" runs once for items a and c
    items = response["items"]
    assert [item.get("id") for item in items] == ["a", "b", "c", None]
    assert items[0]["results"] == items[2]["results"]
    assert items[1]["error"]["code"] == "UNSUPPORTED_TASK_TYPE"
    assert items[1]["error"]["path"] == "/items/1/tasks/t/type"
    assert items[3]["usage"]["rows"] == 1
    assert response["model"] == MODEL


@pytest.mark.parametrize(
    ("body", "path"),
    [
        ({"model": MODEL, "items": []}, "/items"),
        ({"model": MODEL, "items": [{"input": "x", "tasks": {}}] * 5}, "/items"),
        ({"model": MODEL, "items": [{"id": "a"}, {"id": "a"}]}, "/items/1/id"),
        ({"model": MODEL, "items": [{"id": 1}]}, "/items/0/id"),
        ({"model": MODEL, "items": [], "extra": 1}, "/extra"),
    ],
)
def test_bad_batch_envelopes_are_rejected(body, path):
    with pytest.raises(APIError) as info:
        make_engine().process_batch(body, FakeModel().run_rows)

    assert info.value.path == path


def test_runner_failure_is_reported_as_internal_error():
    def broken(rows):
        raise RuntimeError("boom")

    with pytest.raises(APIError) as info:
        make_engine().process({"model": MODEL, "input": "x", "tasks": {"t": {"type": "span", "labels": "a"}}}, broken)

    assert (info.value.code, info.value.status) == ("INTERNAL", 500)


def test_async_path_dispatches_rows_concurrently():
    model = FakeModel()
    in_flight = []

    async def run_row(row):
        in_flight.append(row["labels"])
        await asyncio.sleep(0)
        return model.run_row(row)

    body = {
        "model": MODEL,
        "input": "Alice in Paris",
        "tasks": {"a": {"type": "span", "labels": "person"}, "b": {"type": "span", "labels": "location"}},
        "options": {"include_usage": True},
    }
    response = asyncio.run(make_engine().process_async(body, run_row))

    assert in_flight == [["person"], ["location"]]
    assert response["results"]["a"]["value"][0]["text"] == "Alice"
    assert response["results"]["b"]["value"][0]["text"] == "Paris"
    assert response["usage"]["rows"] == 2


def test_api_errors_survive_pickling():
    import pickle  # noqa: PLC0415

    error = pickle.loads(pickle.dumps(APIError("UNKNOWN_ADAPTER", "gone", "/options/adapter_id", {"a": 1})))

    assert (error.code, error.message, error.path, error.details) == (
        "UNKNOWN_ADAPTER",
        "gone",
        "/options/adapter_id",
        {"a": 1},
    )


# -- server glue ---------------------------------------------------------------------


def test_run_rows_batch_on_relation_model_without_relations_returns_one_result_per_text():
    server = GLiNERServer.__new__(GLiNERServer)
    server.config = Mock(max_labels=-1)
    server._supports_relations = True
    server._run_batch_internal = Mock(return_value=([["e1"], ["e2"], ["e3"]], [[], [], []]))

    results = server.run_rows_batch(["a", "b", "c"], ["person"], None, 0.5, 0.5, True, False, None)

    assert results == [{"entities": ["e1"]}, {"entities": ["e2"]}, {"entities": ["e3"]}]


def test_server_run_rows_feeds_per_row_values_to_one_model_batch():
    server = GLiNERServer.__new__(GLiNERServer)
    server.config = Mock(max_labels=-1, max_batch_size=32, calibration_min_seq_len=1, max_model_len=512)
    server._supports_relations = True
    server.batch_size_fn = lambda seq_len=None: 32
    calls = []

    def run_batch_internal(texts, labels, relations=None, **kwargs):
        calls.append((texts, labels, relations, kwargs))
        return [[] for _ in texts], [[] for _ in texts]

    server._run_batch_internal = run_batch_internal
    rows = [
        {"text": "a", "labels": ["x"], "relations": None, "threshold": 0.1, "relation_threshold": 0.5,
         "flat_ner": True, "multi_label": False, "adapter_id": None},
        {"text": "b", "labels": {"y": "why"}, "relations": ["r"], "threshold": 0.2, "relation_threshold": 0.7,
         "flat_ner": False, "multi_label": True, "adapter_id": "legal"},
    ]  # fmt: skip
    outputs = server.run_rows(rows)

    assert len(calls) == 1
    texts, labels, relations, kwargs = calls[0]
    assert texts == ["a", "b"]
    assert labels == [["x"], {"y": "why"}]
    assert relations == [[], ["r"]]
    assert kwargs == {
        "threshold": [0.1, 0.2],
        "relation_threshold": [0.5, 0.7],
        "flat_ner": [True, False],
        "multi_label": [False, True],
        "adapter_ids": [None, "legal"],
    }
    assert outputs == [{"entities": [], "relations": []}, {"entities": [], "relations": []}]


def test_rows_beyond_the_batch_size_run_in_precompiled_batch_sizes():
    """19 rows with batch size 8 and compilation: GPU batches of 8, 8 and 4 (3 rows + 1 filler)."""
    server = GLiNERServer.__new__(GLiNERServer)
    server.config = Mock(
        max_labels=-1,
        max_batch_size=8,
        enable_compilation=True,
        precompiled_batch_sizes=[1, 2, 4, 8],
        calibration_min_seq_len=1,
        max_model_len=512,
    )
    server._supports_relations = False
    server.model = Mock(is_runtime_model=False)
    server.batch_size_fn = lambda seq_len=None: 8
    gpu_batches = []

    def run_batch_ner(texts, labels, threshold, flat_ner, multi_label, adapter_ids):
        gpu_batches.append(len(texts))
        return [[{"label": labels[i][0]}] for i in range(len(texts))]

    server._run_batch_ner = run_batch_ner
    rows = [
        {"text": f"t{i}", "labels": [f"l{i}"], "relations": None, "threshold": 0.5, "relation_threshold": 0.5,
         "flat_ner": True, "multi_label": False, "adapter_id": None}
        for i in range(19)
    ]  # fmt: skip
    outputs = server.run_rows(rows)

    assert gpu_batches == [8, 8, 4]
    assert [output["entities"][0]["label"] for output in outputs] == [f"l{i}" for i in range(19)]
    stats = server.batch_stats.to_dict()
    assert stats["rows_per_pass"] == {"3": 1, "8": 2}
    assert stats["gpu_rows_per_pass"] == {"4": 1, "8": 2}


# -- defaults and per-task settings ---------------------------------------------------


def _known_adapters(*known):
    def resolve(adapter_id):
        if adapter_id is not None and adapter_id not in known:
            raise KeyError(f"Unknown LoRA adapter id: {adapter_id}")

    return resolve


def test_defaults_apply_to_tasks_that_do_not_set_their_own():
    engine = make_engine(resolve_adapter=_known_adapters("news", "legal"))
    tasks = {
        "a": {"type": "span", "labels": ["person"]},
        "b": {"type": "span", "labels": ["location"], "threshold": 0.7, "adapter_id": "legal", "flat_ner": True},
        "c": {"type": "relation", "labels": ["works at"], "entity_labels": ["person", "organization"]},
    }
    defaults = {"threshold": 0.3, "adapter_id": "news", "flat_ner": False, "multi_label": True, "entity_threshold": 0.2}
    response, model = process(engine, "Alice in Paris works at Acme", tasks, defaults=defaults)

    rows = model.calls[0]
    assert [row["adapter_id"] for row in rows] == ["news", "legal", "news"]
    assert [row["threshold"] for row in rows] == [0.3, 0.7, 0.2]  # relation rows use entity_threshold here
    assert rows[2]["relation_threshold"] == 0.3  # the general threshold is the relation cutoff
    assert [row["flat_ner"] for row in rows] == [False, True, False]
    assert all(row["multi_label"] for row in rows)
    results = response["results"]
    assert (results["a"]["threshold"], results["a"]["adapter_id"]) == (0.3, "news")
    assert (results["b"]["threshold"], results["b"]["adapter_id"]) == (0.7, "legal")
    assert (results["c"]["threshold"], results["c"]["entity_threshold"]) == (0.3, 0.2)


def test_null_in_a_task_skips_the_default():
    engine = make_engine(resolve_adapter=_known_adapters("news"))
    tasks = {"a": {"type": "span", "labels": ["person"], "threshold": None, "adapter_id": None}}
    response, model = process(engine, "Alice", tasks, defaults={"threshold": 0.3, "adapter_id": "news"})

    assert (model.calls[0][0]["threshold"], model.calls[0][0]["adapter_id"]) == (0.5, None)
    assert (response["results"]["a"]["threshold"], response["results"]["a"]["adapter_id"]) == (0.5, None)


def test_tasks_with_different_adapters_get_their_own_rows():
    engine = make_engine(resolve_adapter=_known_adapters("a", "b"))
    tasks = {
        "base": {"type": "span", "labels": ["person"]},
        "with_a": {"type": "span", "labels": ["person"], "adapter_id": "a"},
        "also_a": {"type": "span", "labels": ["person"], "adapter_id": "a"},
        "with_b": {"type": "span", "labels": ["person"], "adapter_id": "b"},
    }
    _, model = process(engine, "Alice", tasks)

    assert [row["adapter_id"] for row in model.calls[0]] == [None, "a", "b"]  # "also_a" shares the "a" row


@pytest.mark.parametrize(
    ("body_extra", "task_extra", "path"),
    [
        ({}, {"adapter_id": "missing"}, "/tasks/t/adapter_id"),
        ({"defaults": {"adapter_id": "missing"}}, {}, "/defaults/adapter_id"),
        ({"options": {"adapter_id": "missing"}}, {}, "/options/adapter_id"),
    ],
)
def test_unknown_adapter_error_points_to_where_it_was_set(body_extra, task_extra, path):
    body = {"model": MODEL, "input": "x", "tasks": {"t": {"type": "span", "labels": "a", **task_extra}}, **body_extra}
    error = error_of(make_engine(resolve_adapter=_known_adapters()), body)

    assert (error.code, error.path) == ("UNKNOWN_ADAPTER", path)


def test_options_adapter_id_is_an_alias_of_the_default_adapter():
    engine = make_engine(resolve_adapter=_known_adapters("news", "legal"))
    _, model = process(engine, "Alice", {"t": {"type": "span", "labels": "person"}}, options={"adapter_id": "news"})
    assert model.calls[0][0]["adapter_id"] == "news"

    body = {
        "model": MODEL,
        "input": "x",
        "tasks": {"t": {"type": "span", "labels": "a"}},
        "options": {"adapter_id": "news"},
        "defaults": {"adapter_id": "legal"},
    }
    assert error_of(engine, body).path == "/options/adapter_id"


@pytest.mark.parametrize(
    ("defaults", "code", "path"),
    [
        ({"labels": ["x"]}, "INVALID_REQUEST", "/defaults/labels"),
        ({"threshold": "high"}, "INVALID_REQUEST", "/defaults/threshold"),
        ({"entity_threshold": 3}, "INVALID_REQUEST", "/defaults/entity_threshold"),
        ({"multi_label": "yes"}, "INVALID_REQUEST", "/defaults/multi_label"),
        ({"adapter_id": 5}, "INVALID_REQUEST", "/defaults/adapter_id"),
        ({"flat": True, "flat_ner": False}, "INVALID_REQUEST", "/defaults/flat_ner"),
        ({"instruction": "find people"}, "UNSUPPORTED_FIELD", "/defaults/instruction"),
        ("x", "INVALID_REQUEST", "/defaults"),
    ],
)
def test_bad_defaults_are_rejected_with_a_path(defaults, code, path):
    body = {"model": MODEL, "input": "x", "tasks": {"t": {"type": "span", "labels": "a"}}, "defaults": defaults}
    error = error_of(make_engine(), body)

    assert (error.code, error.path) == (code, path)


def test_batch_defaults_apply_to_items_and_item_defaults_override_them():
    engine = make_engine(resolve_adapter=_known_adapters("news", "legal"))
    task = {"t": {"type": "span", "labels": ["person"]}}
    body = {
        "model": MODEL,
        "defaults": {"threshold": 0.3, "adapter_id": "news"},
        "items": [
            {"id": "a", "input": "Alice", "tasks": task},
            {"id": "b", "input": "Bob", "tasks": task, "defaults": {"adapter_id": "legal"}},
            {"id": "c", "input": "Alice", "tasks": task, "defaults": {"adapter_id": "missing"}},
        ],
    }
    model = FakeModel()
    response = engine.process_batch(body, model.run_rows)

    assert [(row["text"], row["threshold"], row["adapter_id"]) for row in model.calls[0]] == [
        ("Alice", 0.3, "news"),
        ("Bob", 0.3, "legal"),
    ]
    items = response["items"]
    assert items[0]["results"]["t"]["adapter_id"] == "news"
    assert items[2]["error"]["code"] == "UNKNOWN_ADAPTER"
    assert items[2]["error"]["path"] == "/items/2/defaults/adapter_id"
