"""Concurrent traffic on both GLiNER serving APIs: ``POST /gliner`` and ``/gliner/v1/process``.

Two levels:

* **Engine, no model.** A simulated dynamic batcher merges rows from many
  in-flight process requests into shared batches, the way Ray Serve's
  ``@serve.batch`` does. Every response must equal the same request run alone.
* **Ray Serve over HTTP, real model.** Threads and async clients send a mix of
  legacy and process requests (single and multi-task, structured input,
  batch requests, invalid requests) at the same time. Every response must
  match the same request sent alone, both APIs must agree with each other,
  and errors must stay with the request that caused them. Needs Ray, a CUDA
  GPU and ``knowledgator/gliner-relex-base-v1.0``; takes about a minute.
"""

import json
import random
import socket
import asyncio
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch
import pytest

from tests.test_serve_process import MODEL, FakeModel, make_engine

# -- engine with a simulated dynamic batcher -----------------------------------------


class SimulatedBatcher:
    """Collects rows from concurrent callers and runs them in shared batches."""

    def __init__(self, model, max_batch_size=16, wait_s=0.002):
        self.model = model
        self.max_batch_size = max_batch_size
        self.wait_s = wait_s
        self.pending = []
        self.batch_sizes = []
        self.flusher = None

    async def run_row(self, row):
        future = asyncio.get_running_loop().create_future()
        self.pending.append((row, future))
        if len(self.pending) >= self.max_batch_size:
            self.flush()
        elif self.flusher is None:
            self.flusher = asyncio.get_running_loop().call_later(self.wait_s, self.flush)
        return await future

    def flush(self):
        if self.flusher is not None:
            self.flusher.cancel()
            self.flusher = None
        batch, self.pending = self.pending[: self.max_batch_size], self.pending[self.max_batch_size :]
        if not batch:
            return
        self.batch_sizes.append(len(batch))
        outputs = self.model.run_rows([row for row, _ in batch])
        for (_, future), output in zip(batch, outputs, strict=True):
            future.set_result(output)
        if self.pending:
            self.flusher = asyncio.get_running_loop().call_later(self.wait_s, self.flush)


def _fake_requests():
    names = ["Alice", "Bob"]
    places = ["London", "Paris"]
    requests = []
    for index in range(40):
        text = f"{names[index % 2]} met {names[(index + 1) % 2]} at Acme in {places[index % 2]} #{index}."
        kind = index % 5
        if kind == 0:
            tasks = {"p": {"type": "span", "labels": ["person"], "threshold": 0.3 + index / 100}}
        elif kind == 1:
            tasks = {
                "p": {"type": "span", "labels": ["person"]},
                "l": {"type": "span", "labels": {"location": {"title": "place"}}, "flat_ner": False},
                "r": {"type": "relation", "labels": ["works at"], "entity_labels": ["person", "organization"]},
            }
        elif kind == 2:
            body = {
                "model": MODEL,
                "items": [
                    {"id": "a", "input": text, "tasks": {"o": {"type": "span", "labels": "organization"}}},
                    {"id": "b", "input": {"note": text}, "tasks": {"p": {"type": "span", "labels": ["person"]}}},
                ],
            }
            requests.append(("batch", body))
            continue
        elif kind == 3:
            tasks = {"bad": {"type": "classify", "labels": ["x"]}}
        else:
            tasks = {"p": {"type": "span", "labels": ["person", "location"], "multi_label": True}}
        requests.append(("process", {"model": MODEL, "input": {"t": text, "n": index}, "tasks": tasks}))
    return requests


def _run_alone(engine, kind, body):
    model = FakeModel()
    if kind == "batch":
        return engine.process_batch(body, model.run_rows)
    outcome = engine.process_many([body], model.run_rows)[0]
    return outcome.to_dict() if hasattr(outcome, "to_dict") else outcome


def test_concurrent_requests_share_batches_and_get_their_own_results():
    engine = make_engine()
    requests = _fake_requests()
    expected = [_run_alone(engine, kind, body) for kind, body in requests]
    batcher = SimulatedBatcher(FakeModel())

    async def send(kind, body):
        await asyncio.sleep(random.random() * 0.003)  # interleave arrivals
        if kind == "batch":
            return await engine.process_batch_async(body, batcher.run_row)
        outcome = (await engine.process_many_async([body], batcher.run_row))[0]
        return outcome.to_dict() if hasattr(outcome, "to_dict") else outcome

    async def main():
        return await asyncio.gather(*(send(kind, body) for kind, body in requests))

    random.seed(0)
    results = asyncio.run(main())

    assert results == expected
    # Rows from different requests ran together: no single request has more than 3 rows.
    assert max(batcher.batch_sizes) > 3


# -- Ray Serve over HTTP --------------------------------------------------------------

SERVE_MODEL = "knowledgator/gliner-relex-base-v1.0"
ENTITY_LABELS = ["person", "organization", "location"]
RELATIONS = ["founded", "works at", "headquartered in"]
TEXTS = [
    "Bill Gates founded Microsoft. The company is headquartered in Redmond.",
    "Alice Smith works at Google in London.",
    "Steve Jobs founded Apple in Cupertino.",
    "Bob Jones works at Amazon, which is headquartered in Seattle.",
    "Elon Musk founded SpaceX in Hawthorne.",
    "Carol White works at IBM in Paris.",
]
# Rows batched with other rows are padded differently, which moves scores slightly.
SCORE_TOL = 0.02


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


MAX_BATCH = 8

_SERVED = {}


@pytest.fixture(scope="module")
def server_url():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    pytest.importorskip("ray")
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    try:
        snapshot_download(SERVE_MODEL)
    except Exception as exc:
        pytest.skip(f"cannot download {SERVE_MODEL}: {exc}")

    import ray  # noqa: PLC0415

    from gliner.serve import GLiNERServeConfig, serve, shutdown  # noqa: PLC0415

    port = _free_port()
    config = GLiNERServeConfig(
        model=SERVE_MODEL,
        dtype="float32",
        max_model_len=384,
        enable_compilation=False,
        batch_wait_timeout_ms=10.0,
        max_batch_size=MAX_BATCH,
        precompiled_batch_sizes=[1, 2, 4, MAX_BATCH],
        http_port=port,
        enable_polylora=False,
    )
    _SERVED["handle"] = serve(config)
    yield f"http://127.0.0.1:{port}"
    shutdown()
    if ray.is_initialized():
        ray.shutdown()


def _post(url, body, *, raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _legacy(text, **params):
    return ("legacy", {"text": text, "labels": ENTITY_LABELS, **params})


def _process(input, tasks):
    return ("process", {"model": SERVE_MODEL, "input": input, "tasks": tasks})


def _workload():
    """A mix of legacy and process requests, each paired with a key to compare responses by."""
    work = []
    for index, text in enumerate(TEXTS):
        threshold = 0.3 + 0.1 * (index % 3)
        work += [
            _legacy(text, threshold=threshold),
            _legacy(text, relations=RELATIONS, threshold=0.4, relation_threshold=0.5),
            _legacy(text, flat_ner=False, multi_label=True),
            _legacy(text, labels={"person": "person", "organization": "company"}),
            _process(text, {"ents": {"type": "span", "labels": ENTITY_LABELS, "threshold": threshold}}),
            _process(
                text,
                {
                    "people": {"type": "span", "labels": "person", "threshold": 0.3},
                    "orgs": {"type": "span", "labels": {"organization": {"title": "company"}}, "flat_ner": False},
                    "rels": {
                        "type": "relation",
                        "labels": RELATIONS,
                        "entity_labels": ENTITY_LABELS,
                        "threshold": 0.5,
                        "entity_threshold": 0.4,
                    },
                },
            ),
            _process(
                {"headline": text, "meta": {"source": "wire", "tags": ["business"]}},
                {"ents": {"type": "span", "labels": ENTITY_LABELS, "multi_label": True}},
            ),
            (
                "batch",
                {
                    "model": SERVE_MODEL,
                    "items": [
                        {"id": "same-1", "input": text, "tasks": {"e": {"type": "span", "labels": ENTITY_LABELS}}},
                        {"id": "same-2", "input": text, "tasks": {"e": {"type": "span", "labels": ENTITY_LABELS}}},
                        {"id": "bad", "input": text, "tasks": {"e": {"type": "json", "labels": ["x"]}}},
                    ],
                },
            ),
        ]
    # Shared settings in ``defaults``, overridden per task.
    for text in TEXTS:
        work.append(
            (
                "process",
                {
                    "model": SERVE_MODEL,
                    "input": text,
                    "defaults": {"threshold": 0.35, "flat_ner": False, "entity_threshold": 0.4},
                    "tasks": {
                        "ents": {"type": "span", "labels": ENTITY_LABELS},
                        "strict": {"type": "span", "labels": ENTITY_LABELS, "threshold": 0.8, "flat_ner": True},
                        "rels": {"type": "relation", "labels": RELATIONS, "entity_labels": ENTITY_LABELS},
                    },
                },
            )
        )
    # Invalid requests interleaved with valid ones must not disturb them.
    work += [
        _process("x", {"t": {"type": "classify", "labels": ["a"]}}),
        _process("x", {"t": {"type": "span", "labels": "a", "instruction": "find a"}}),
        ("process", {"model": "nope", "input": "x", "tasks": {"t": {"type": "span", "labels": "a"}}}),
        ("process", {"model": SERVE_MODEL, "input": "x", "tasks": {"t": {"type": "span", "labels": "a"}},
                     "options": {"adapter_id": "missing"}}),
        ("process", {"model": SERVE_MODEL, "input": "x",
                     "tasks": {"ok": {"type": "span", "labels": "a"},
                               "bad": {"type": "span", "labels": "a", "adapter_id": "missing"}}}),
        ("process", {"model": SERVE_MODEL, "input": "x", "tasks": {"t": {"type": "span", "labels": "a"}},
                     "defaults": {"threshold": "high"}}),
        ("process-raw", b"{not json"),
        ("old-route", {"model": SERVE_MODEL, "input": "x", "tasks": {"t": {"type": "span", "labels": "a"}}}),
        ("legacy", {"text": TEXTS[0], "labels": ENTITY_LABELS, "adapter_id": "missing"}),
    ]  # fmt: skip
    return work


def _send(base, item):
    kind, body = item
    if kind == "legacy":
        return _post(f"{base}/gliner", body)
    if kind == "old-route":  # the API was renamed from analyze to process
        return _post(f"{base}/gliner/v1/analyze", body)
    if kind == "process":
        return _post(f"{base}/gliner/v1/process", body)
    if kind == "batch":
        return _post(f"{base}/gliner/v1/process/batch", body)
    return _post(f"{base}/gliner/v1/process", None, raw=body)


def _spans_equal(got, ref, threshold):
    """Same spans, scores within tolerance; a span may only be missing on one side near the threshold."""

    def by_key(spans):
        return {(s["start"], s["end"], s["label"], s.get("source", "")): s["score"] for s in spans}

    got, ref = by_key(got), by_key(ref)
    for key in got.keys() | ref.keys():
        if key in got and key in ref:
            if abs(got[key] - ref[key]) > SCORE_TOL:
                return False
        elif abs(got.get(key, ref.get(key)) - threshold) > SCORE_TOL:
            return False
    return True


def _relations_equal(got, ref, threshold):
    def by_key(relations):
        return {
            (r["head"]["start"], r["head"]["end"], r["relation"], r["tail"]["start"], r["tail"]["end"]): r["score"]
            for r in relations
        }

    got, ref = by_key(got), by_key(ref)
    return all(
        (key in got and key in ref and abs(got[key] - ref[key]) <= SCORE_TOL)
        or abs(got.get(key, ref.get(key, threshold)) - threshold) <= SCORE_TOL
        for key in got.keys() | ref.keys()
    )


def _results_equal(got, ref):
    if set(got) != set(ref):
        return False
    for name, result in got.items():
        other = ref[name]
        if result["type"] != other["type"] or result.get("threshold") != other.get("threshold"):
            return False
        compare = _relations_equal if result["type"] == "relation" else _spans_equal
        if not compare(result["value"], other["value"], result["threshold"]):
            return False
    return True


def _responses_equal(item, got, ref):
    kind, body = item
    (got_status, got_body), (ref_status, ref_body) = got, ref
    if got_status != ref_status:
        return False
    if "error" in ref_body:
        return got_body == ref_body
    if kind == "legacy":
        threshold = body.get("threshold", 0.5)
        return _spans_equal(got_body["entities"], ref_body["entities"], threshold) and _relations_equal(
            got_body.get("relations", []), ref_body.get("relations", []), body.get("relation_threshold", 0.5)
        )
    if kind == "batch":
        return len(got_body["items"]) == len(ref_body["items"]) and all(
            g.get("id") == r.get("id")
            and (g == r if "error" in r else "results" in g and _results_equal(g["results"], r["results"]))
            for g, r in zip(got_body["items"], ref_body["items"], strict=True)
        )
    return _results_equal(got_body["results"], ref_body["results"])


def _mismatch_summary(item, got, ref):
    def brief(response):
        status, body = response
        errors = [body["error"]] if "error" in body else [i["error"] for i in body.get("items", []) if "error" in i]
        return status, errors

    return item[0], json.dumps(item[1])[:120], "concurrent:", brief(got), "alone:", brief(ref)


def test_concurrent_mixed_traffic_matches_requests_sent_alone(server_url):
    work = _workload()
    alone = [_send(server_url, item) for item in work]

    rng = random.Random(0)
    shuffled = [index for index in range(len(work)) for _ in range(3)]  # every request three times
    rng.shuffle(shuffled)
    with ThreadPoolExecutor(48) as pool:
        concurrent = list(pool.map(lambda index: _send(server_url, work[index]), shuffled))

    mismatches = [
        (work[index], got, alone[index])
        for index, got in zip(shuffled, concurrent, strict=True)
        if not _responses_equal(work[index], got, alone[index])
    ]
    assert not mismatches, [_mismatch_summary(*mismatch) for mismatch in mismatches[:5]]

    statuses = {(work[index][0], got[0]) for index, got in zip(shuffled, concurrent, strict=True)}
    assert ("process", 422) in statuses and ("process", 404) in statuses and ("legacy", 404) in statuses
    assert ("old-route", 404) in statuses
    assert {status for kind, status in statuses if kind == "batch"} == {200}


def test_both_apis_agree_on_the_same_question_under_load(server_url):
    """Legacy and process requests for the same model input, sent concurrently, return the same predictions."""
    relation_task = {
        "type": "relation",
        "labels": RELATIONS,
        "entity_labels": ENTITY_LABELS,
        "threshold": 0.5,
        "entity_threshold": 0.4,
    }
    pairs = []
    for text in TEXTS:
        span_task = {"type": "span", "labels": ENTITY_LABELS, "threshold": 0.4}
        relex = _legacy(text, relations=RELATIONS, threshold=0.4, relation_threshold=0.5)
        pairs += [(_legacy(text, threshold=0.4), _process(text, {"t": span_task}))] * 4
        pairs += [(relex, _process(text, {"t": relation_task}))] * 4

    with ThreadPoolExecutor(32) as pool:
        legacy_out = pool.map(lambda pair: _send(server_url, pair[0]), pairs)
        process_out = pool.map(lambda pair: _send(server_url, pair[1]), pairs)
        responses = list(zip(pairs, legacy_out, process_out, strict=True))

    for (legacy_request, _), (legacy_status, legacy_body), (process_status, process_body) in responses:
        assert legacy_status == process_status == 200
        result = process_body["results"]["t"]
        if "relations" in legacy_request[1]:
            assert result["value"], "expected relations"
            assert _relations_equal(result["value"], legacy_body["relations"], 0.5)
        else:
            assert _spans_equal(result["value"], legacy_body["entities"], 0.4)


def test_async_clients_on_both_apis(server_url):
    from gliner.serve import GLiNERClient  # noqa: PLC0415

    client = GLiNERClient(server_url, timeout=120)
    tasks = {"ents": {"type": "span", "labels": ENTITY_LABELS}}
    alone_process = [client.process(text, tasks) for text in TEXTS]
    alone_predict = [client.predict(text, ENTITY_LABELS) for text in TEXTS]

    async def main():
        process = [client.process_async(text, tasks) for text in TEXTS * 4]
        predict = [client.predict_async(text, ENTITY_LABELS) for text in TEXTS * 4]
        batch = client.process_batch_async([{"id": str(i), "input": t, "tasks": tasks} for i, t in enumerate(TEXTS)])
        return await asyncio.gather(asyncio.gather(*process), asyncio.gather(*predict), batch)

    process, predict, batch = asyncio.run(main())

    for index, response in enumerate(process):
        assert _results_equal(response["results"], alone_process[index % len(TEXTS)]["results"])
    for index, response in enumerate(predict):
        assert _spans_equal(response["entities"], alone_predict[index % len(TEXTS)]["entities"], 0.5)
    for index, item in enumerate(batch["items"]):
        assert item["id"] == str(index)
        assert _results_equal(item["results"], alone_process[index]["results"])


def test_concurrency_batches_requests(server_url):
    """Many concurrent requests finish faster than the same requests sent one at a time."""
    import time  # noqa: PLC0415

    work = [_process(text, {"ents": {"type": "span", "labels": ENTITY_LABELS}}) for text in TEXTS * 8]
    work += [_legacy(text) for text in TEXTS * 8]

    start = time.perf_counter()
    for item in work:
        _send(server_url, item)
    sequential = time.perf_counter() - start

    start = time.perf_counter()
    with ThreadPoolExecutor(48) as pool:
        statuses = [status for status, _ in pool.map(lambda item: _send(server_url, item), work)]
    concurrent = time.perf_counter() - start

    assert statuses == [200] * len(work)
    assert concurrent < sequential, (concurrent, sequential)


def test_many_task_requests_run_in_gpu_batches_of_bounded_size(server_url):
    """Requests with more tasks than the batch size are split; rows of all requests share capped batches."""
    handle = _SERVED["handle"]
    num_tasks = 20  # each request alone needs more than two full batches
    work = [
        _process(
            text,
            {
                f"t{i}": {"type": "span", "labels": [ENTITY_LABELS[i % 3]], "threshold": 0.3 + i / 100}
                for i in range(num_tasks)
            },
        )
        for text in TEXTS
    ]
    work += [_legacy(text, threshold=0.3) for text in TEXTS]  # legacy rows join the same batches
    alone = [_send(server_url, item) for item in work]

    handle.batch_stats.remote(reset=True).result()
    with ThreadPoolExecutor(len(work)) as pool:
        concurrent = list(pool.map(lambda item: _send(server_url, item), work))
    stats = handle.batch_stats.remote().result()

    for item, got, ref in zip(work, concurrent, alone, strict=True):
        assert _responses_equal(item, got, ref), _mismatch_summary(item, got, ref)
    assert stats["rows"] == len(TEXTS) * (num_tasks + 1)
    assert stats["max_rows"] <= MAX_BATCH
    assert stats["forward_passes"] < len(TEXTS) * -(-num_tasks // MAX_BATCH) + len(
        TEXTS
    )  # rows of different requests merged
