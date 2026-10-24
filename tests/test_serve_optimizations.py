"""GLiNER serve with torch.compile, FlashDeBERTa, PolyLoRA and FP8.

Each test starts a ``GLiNERServer`` (the model side of the Ray Serve deployment;
Ray itself is not needed) on ``gliner-community/gliner_small-v2.5`` and checks it
against an fp32 eager Hugging Face reference. Adapters are applied to
the reference with PEFT, so PolyLoRA outputs are checked against an independent
implementation.

Covered configurations:

- eager DeBERTa + compilation
- eager DeBERTa + PolyLoRA
- FlashDeBERTa + compilation
- FlashDeBERTa + PolyLoRA + compilation + FP8

These tests need a CUDA GPU (FP8: compute capability 8.9+) and take a few minutes,
mostly in torch.compile.
"""

import copy

import torch
import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

MODEL = "gliner-community/gliner_small-v2.5"
TARGET_MODULES = ["query_proj", "key_proj", "value_proj", "dense"]
LABELS = ["person", "organization", "location"]
TEXTS = [
    "John Smith works at Google in Mountain View.",
    "Paris is the capital of France, and Berlin is the capital of Germany.",
    "Angela Merkel met Emmanuel Macron in Brussels on Monday to discuss the budget.",
    "Apple released a new phone.",
]
ADAPTERS = ["legal", "medical"]
ADAPTER_IDS = ["legal", "__base__", "medical", "legal"]  # a mixed batch, with a base-only row

# Logit error limits (max, mean), in units of the reference logit std. The reference
# runs in fp32. The bf16 paths (compiled, FlashDeBERTa, PolyLoRA) measure about
# (0.03, 0.005); FP8 linear layers and FP8 Q·Kᵀ about (0.15, 0.02). The test
# adapters move the logits by well over both limits.
BF16_TOL = (0.1, 0.015)
FP8_TOL = (0.4, 0.06)


def _fp8_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # The server sets USE_FLASHDEBERTA in os.environ; never let it leak into other tests.
    monkeypatch.delenv("USE_FLASHDEBERTA", raising=False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def model_dir():
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    try:
        return snapshot_download(MODEL)
    except Exception as exc:  # offline and not cached
        pytest.skip(f"cannot download {MODEL}: {exc}")


@pytest.fixture(scope="module")
def adapter_cache_dir(model_dir, tmp_path_factory):
    """A PolyLoRA disk cache holding two PEFT adapters for the encoder."""
    peft = pytest.importorskip("peft")
    pytest.importorskip("polylora")
    from polylora.store import DiskAdapterCache  # noqa: PLC0415

    encoder = _load_reference(model_dir).model.token_rep_layer.bert_layer.model
    root = tmp_path_factory.mktemp("adapters")
    cache = DiskAdapterCache(root / "cache")
    for seed, name in enumerate(ADAPTERS):
        torch.manual_seed(100 + seed)
        peft_model = peft.get_peft_model(
            copy.deepcopy(encoder),
            peft.LoraConfig(r=8, lora_alpha=16, target_modules=TARGET_MODULES, lora_dropout=0.0),
        )
        with torch.no_grad():
            for module_name, param in peft_model.named_parameters():
                if "lora_B" in module_name:  # PEFT initialises B to zero: make the adapter do something
                    param.normal_(0.0, 0.06)
        peft_model.save_pretrained(root / name)
        cache.add_adapter(name, root / name)
    return str(root / "cache"), {name: str(root / name) for name in ADAPTERS}


def _load_reference(model_dir):
    """fp32 eager Hugging Face DeBERTa GLiNER (no FlashDeBERTa, no compile)."""
    from gliner import GLiNER  # noqa: PLC0415

    return GLiNER.from_pretrained(model_dir, map_location="cuda", dtype=torch.float32).eval()


def _make_server(model_dir, monkeypatch, **overrides):
    from gliner.serve.config import GLiNERServeConfig  # noqa: PLC0415
    from gliner.serve.server import GLiNERServer  # noqa: PLC0415

    config = {
        "model": model_dir,
        "device": "cuda",
        "dtype": "bfloat16",
        "max_model_len": 384,
        "max_batch_size": 4,
        "precompiled_batch_sizes": [1, 2, 4],
        "warmup_iterations": 1,
        "tokenizer_threads": 0,
        "enable_compilation": False,
        "polylora_adapter_weight_modules": TARGET_MODULES,
    }
    config.update(overrides)
    if not config.get("enable_flashdeberta"):
        monkeypatch.delenv("USE_FLASHDEBERTA", raising=False)
    return GLiNERServer(GLiNERServeConfig(**config))


@torch.inference_mode()
def _logits(model, texts, run_kwargs=None, collator=None):
    """Span logits for ``texts``, through the same low-level path the server uses."""
    prepared = model.prepare_batch(texts, LABELS)
    batch = model.collate_batch(
        prepared["input_x"],
        prepared["entity_types"],
        collator or model.create_collator(),
        label_names=prepared["label_names"],
    )
    output = model.run_batch(batch, threshold=0.5, move_to_device=True, **(run_kwargs or {}))
    return output.logits.float()


def _server_logits(server, texts, adapter_ids=None):
    """Logits through the server's own adapter resolution."""
    resolved = server._resolve_adapter_ids(adapter_ids, list(range(len(texts))))
    return _logits(server.model, texts, server._adapter_run_kwargs(resolved), server.collator)


@torch.inference_mode()
def _reference_rows(reference, texts, adapter_ids=None, adapter_paths=None):
    """Logits of every text on its own, with the row's adapter applied through PEFT."""
    bert_layer = reference.model.token_rep_layer.bert_layer
    base = bert_layer.model
    rows = []
    for index, text in enumerate(texts):
        adapter_id = None if adapter_ids is None else adapter_ids[index]
        if adapter_id not in (None, "__base__"):
            from peft import PeftModel  # noqa: PLC0415

            bert_layer.model = PeftModel.from_pretrained(copy.deepcopy(base), adapter_paths[adapter_id]).cuda()
        try:
            rows.append(_logits(reference, [text])[0])
        finally:
            bert_layer.model = base
    return rows


def _assert_rows_close(actual, rows, tol, what):
    """Compare a batch of logits to per-row references on the valid (unpadded) positions."""
    assert actual.shape[0] == len(rows), f"{what}: batch size {actual.shape[0]} != {len(rows)}"
    assert torch.isfinite(actual).all(), f"{what}: non-finite logits"
    scale = torch.cat([row.flatten() for row in rows]).std().item()
    diffs = []
    for index, row in enumerate(rows):
        got = actual[index, : row.shape[0]]
        assert got.shape == row.shape, f"{what}: row {index} shape {tuple(got.shape)} != {tuple(row.shape)}"
        diffs.append((got - row).abs().flatten())
    diffs = torch.cat(diffs) / scale
    max_err, mean_err = diffs.max().item(), diffs.mean().item()
    print(f"{what}: error max {max_err:.4f}, mean {mean_err:.4f} x logit std")  # noqa: T201
    assert max_err < tol[0], f"{what}: max error {max_err:.4f} x logit std exceeds {tol[0]}"
    assert mean_err < tol[1], f"{what}: mean error {mean_err:.4f} x logit std exceeds {tol[1]}"


def _max_row_diff(rows_a, rows_b):
    scale = torch.cat([row.flatten() for row in rows_b]).std().item()
    diff = max((a - b).abs().max().item() for a, b in zip(rows_a, rows_b, strict=True)) / scale
    print(f"adapter effect: max {diff:.4f} x logit std")  # noqa: T201
    return diff


def _encoder(server):
    model = getattr(server.model.model, "_orig_mod", server.model.model)
    return model.token_rep_layer.bert_layer


def _assert_compiled(server):
    """The whole model is compiled, encoder included (it is not excluded with torch.compiler.disable)."""
    assert isinstance(server.model.model, torch._dynamo.eval_frame.OptimizedModule)
    assert not getattr(_encoder(server).forward, "_torchdynamo_disable", False), "encoder is excluded from compile"


# Entities gliner_small-v2.5 finds in TEXTS without adapters.
EXPECTED_ENTITIES = {
    ("John Smith", "person"),
    ("Google", "organization"),
    ("Paris", "location"),
    ("Angela Merkel", "person"),
    ("Brussels", "location"),
    ("Apple", "organization"),
}


def _assert_predict_works(server, adapter_id=None):
    results = server.predict(TEXTS, LABELS, adapter_id=adapter_id)
    assert len(results) == len(TEXTS)
    found = set()
    for text, result in zip(TEXTS, results, strict=True):
        for entity in result["entities"]:
            assert entity["label"] in LABELS
            assert text[entity["start"] : entity["end"]] == entity["text"]
            found.add((entity["text"], entity["label"]))
    if adapter_id is None:  # the test adapters are random and may change predictions
        assert found >= EXPECTED_ENTITIES, f"missing {EXPECTED_ENTITIES - found}"
    return results


def test_eager_model_with_compilation(model_dir, monkeypatch):
    reference = _reference_rows(_load_reference(model_dir), TEXTS)
    server = _make_server(model_dir, monkeypatch, enable_compilation=True)

    assert type(_encoder(server).model).__name__ == "DebertaV2Model"
    _assert_compiled(server)
    _assert_rows_close(_server_logits(server, TEXTS), reference, BF16_TOL, "compiled eager model")
    _assert_predict_works(server)


def test_eager_model_with_polylora(model_dir, adapter_cache_dir, monkeypatch):
    cache_dir, adapter_paths = adapter_cache_dir
    reference_model = _load_reference(model_dir)
    base_reference = _reference_rows(reference_model, TEXTS)
    server = _make_server(model_dir, monkeypatch, enable_polylora=True, polylora_disk_cache_dir=cache_dir)

    assert type(_encoder(server).model).__name__ == "PolyLoraModel"
    assert type(_encoder(server).model.base_model).__name__ == "DebertaV2Model"
    _assert_rows_close(_server_logits(server, TEXTS), base_reference, BF16_TOL, "PolyLoRA base rows")
    # HF DeBERTa projects the shared relative-position embeddings through query/key_proj
    # once per batch, so mixed-adapter batches with adapters on those layers are only
    # supported on FlashDeBERTa (see the PolyLoRA README). The model runs one adapter per
    # batch, and the server splits a mixed batch (as Ray's batcher builds) by adapter.
    assert not server._polylora_mixes_adapters
    mixed = server._run_batch_internal(TEXTS, LABELS, threshold=0.3, adapter_ids=ADAPTER_IDS)
    for text, adapter_id, entities in zip(TEXTS, ADAPTER_IDS, mixed, strict=True):
        single = server._run_batch_internal([text], LABELS, threshold=0.3, adapter_ids=adapter_id)[0]
        assert [(e["text"], e["label"]) for e in entities] == [(e["text"], e["label"]) for e in single]
    for adapter_id in ADAPTERS:
        adapter_ids = [adapter_id] * len(TEXTS)
        actual = _server_logits(server, TEXTS, adapter_ids)
        expected = _reference_rows(reference_model, TEXTS, adapter_ids, adapter_paths)
        _assert_rows_close(actual, expected, BF16_TOL, f"PolyLoRA adapter {adapter_id}")
        # The adapter changes the output by much more than the tolerance.
        assert _max_row_diff(expected, base_reference) > 3 * BF16_TOL[0]
    _assert_predict_works(server)
    _assert_predict_works(server, adapter_id="medical")
    assert server.adapter_cache_status("medical")["gpu_resident"]


def test_flash_model_with_compilation(model_dir, monkeypatch):
    pytest.importorskip("flashdeberta")
    reference = _reference_rows(_load_reference(model_dir), TEXTS)
    server = _make_server(model_dir, monkeypatch, enable_flashdeberta=True, enable_compilation=True)

    assert type(_encoder(server).model).__name__ == "FlashDebertaV2Model"
    _assert_compiled(server)
    _assert_rows_close(_server_logits(server, TEXTS), reference, BF16_TOL, "compiled FlashDeBERTa")
    _assert_predict_works(server)


@pytest.mark.skipif(not _fp8_supported(), reason="FP8 needs compute capability 8.9+")
def test_flash_model_with_polylora_compilation_and_fp8(model_dir, adapter_cache_dir, monkeypatch):
    pytest.importorskip("flashdeberta")
    pytest.importorskip("torchao")
    cache_dir, adapter_paths = adapter_cache_dir
    reference_model = _load_reference(model_dir)
    reference = _reference_rows(reference_model, TEXTS, ADAPTER_IDS, adapter_paths)
    # The adapters change the output by more than the FP8 limit, so a dropped adapter fails.
    assert _max_row_diff(reference, _reference_rows(reference_model, TEXTS)) > 2 * FP8_TOL[0]
    server = _make_server(
        model_dir,
        monkeypatch,
        enable_flashdeberta=True,
        enable_polylora=True,
        enable_compilation=True,
        quantization="fp8",
        polylora_disk_cache_dir=cache_dir,
    )

    from polylora.layers import MultiLoraLinear  # noqa: PLC0415

    wrapped = _encoder(server).model
    assert type(wrapped).__name__ == "PolyLoraModel"
    assert type(wrapped.base_model).__name__ == "FlashDebertaV2Model"
    _assert_compiled(server)
    lora_layers = [m for m in wrapped.modules() if isinstance(m, MultiLoraLinear)]
    assert lora_layers
    for layer in lora_layers:  # FP8 base layers; adapters stay in the activation dtype
        assert "Float8" in type(layer.base_layer.weight).__name__, type(layer.base_layer.weight)
    attention = [m for m in wrapped.modules() if type(m).__name__ == "FlashDisentangledSelfAttention"]
    assert attention and all(m.attention_precision == "fp8_qk" for m in attention)

    actual = _server_logits(server, TEXTS, ADAPTER_IDS)
    _assert_rows_close(actual, reference, FP8_TOL, "Flash + PolyLoRA + compile + FP8, mixed adapters")
    # New adapter mixes reuse the compiled graphs (no recompile) and stay correct.
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    for mix in (["medical", "legal", "__base__", "medical"], ["__base__", "medical", "medical", "legal"]):
        expected = _reference_rows(reference_model, TEXTS, mix, adapter_paths)
        _assert_rows_close(_server_logits(server, TEXTS, mix), expected, FP8_TOL, f"adapter mix {mix}")
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs, "a new adapter mix recompiled"
    _assert_predict_works(server)
    _assert_predict_works(server, adapter_id="legal")


@pytest.mark.skipif(not _fp8_supported(), reason="FP8 needs compute capability 8.9+")
def test_ray_serve_flash_polylora_compilation_fp8(model_dir, adapter_cache_dir):
    """The full stack behind Ray Serve: HTTP, dynamic batching of mixed-adapter requests."""
    pytest.importorskip("flashdeberta")
    pytest.importorskip("torchao")
    ray = pytest.importorskip("ray")
    pytest.importorskip("ray.serve")
    import socket  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from gliner.serve import GLiNERServeConfig, serve, shutdown  # noqa: PLC0415
    from gliner.serve.client import GLiNERClient, GLiNERClientError  # noqa: PLC0415

    cache_dir, _ = adapter_cache_dir
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    torch.cuda.empty_cache()
    # Replicas must import this checkout of gliner.
    repo_root = str(Path(__file__).resolve().parents[1])
    ray.init(num_cpus=4, include_dashboard=False, runtime_env={"env_vars": {"PYTHONPATH": repo_root}})
    try:
        serve(
            GLiNERServeConfig(
                model=model_dir,
                max_model_len=384,
                max_batch_size=8,
                precompiled_batch_sizes=[1, 2, 4, 8],
                warmup_iterations=1,
                enable_flashdeberta=True,
                enable_polylora=True,
                enable_compilation=True,
                quantization="fp8",
                polylora_adapter_weight_modules=TARGET_MODULES,
                polylora_disk_cache_dir=cache_dir,
                http_port=port,
            )
        )
        client = GLiNERClient(base_url=f"http://localhost:{port}", max_concurrency=16)

        def entities(result):
            return sorted((e["text"], e["label"]) for e in result["entities"])

        base = client.predict(TEXTS, LABELS)
        found = {entity for result in base for entity in entities(result)}
        assert found >= EXPECTED_ENTITIES, f"missing {EXPECTED_ENTITIES - found}"

        # Adapters reach the model: scores move.
        legal = client.predict(TEXTS, LABELS, adapter_id="legal")
        base_scores = [e["score"] for r in base for e in r["entities"]]
        legal_scores = [e["score"] for r in legal for e in r["entities"]]
        assert base_scores != legal_scores, "the adapter did not change any score"

        # Concurrent requests with different adapters are batched together by Ray and
        # must give the same entities as one request at a time.
        jobs = [(TEXTS[i % len(TEXTS)], [None, "legal", "medical"][i % 3]) for i in range(48)]
        with ThreadPoolExecutor(16) as pool:
            concurrent = list(pool.map(lambda job: client.predict(job[0], LABELS, adapter_id=job[1]), jobs))
        one_by_one = {job: client.predict(job[0], LABELS, adapter_id=job[1]) for job in set(jobs)}
        for job, result in zip(jobs, concurrent, strict=True):
            assert entities(result) == entities(one_by_one[job]), job

        status = client.adapter_cache_status("legal")
        assert status["enabled"] and status["cpu_resident"] and status["gpu_resident"]
        with pytest.raises(GLiNERClientError, match="404"):
            client.predict(TEXTS[0], LABELS, adapter_id="unknown")
    finally:
        shutdown()
        ray.shutdown()
