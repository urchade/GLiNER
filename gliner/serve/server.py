"""Ray Serve deployment for GLiNER with dynamic batching and memory-aware batch sizing."""

import os
import logging
from typing import Any, Dict, List, Tuple, Union, Optional

import torch

from .config import GLiNERServeConfig
from .memory import GLiNERMemoryEstimator

logger = logging.getLogger(__name__)


class GLiNERServer:
    """GLiNER Ray Serve deployment with dynamic batching.

    Supports both entity extraction (NER) and relation extraction.
    Automatically detects model type and adjusts behavior accordingly.

    Uses low-level batch methods (prepare_batch, collate_batch, run_batch,
    decode_batch) to avoid DataLoader initialization overhead on each call.

    Features:
        - Dynamic batching with Ray Serve's @serve.batch
        - Memory-aware batch size estimation to prevent CUDA OOM
        - Precompilation for power-of-two batch sizes
        - Support for both NER and relation extraction models
        - FlashDeBERTa support for faster inference
        - Sequence packing for improved throughput
    """

    def __init__(self, config: GLiNERServeConfig):
        """Initialize the GLiNER server deployment.

        Args:
            config: Server configuration with model and serving parameters.
        """
        from gliner import GLiNER, InferencePackingConfig  # noqa: PLC0415

        self.config = config

        env_vars = config.to_env_vars()
        for key, value in env_vars.items():
            os.environ[key] = value

        if config.tokenizer_threads > 0:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            torch.set_num_threads(config.tokenizer_threads)

        torch.set_float32_matmul_precision("high")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map.get(config.dtype.lower(), torch.bfloat16)

        self.memory_estimator = GLiNERMemoryEstimator(
            safety_factor=config.memory_overhead_factor,
            target_memory_fraction=config.target_memory_fraction,
            calibration_probe_batch_size=config.calibration_probe_batch_size,
        )

        if torch.cuda.is_available():
            self.memory_estimator.measure_cuda_context()

        logger.info("Loading model: %s", config.model)
        if config.enable_flashdeberta:
            logger.info("FlashDeBERTa enabled")

        self.model = GLiNER.from_pretrained(
            config.model,
            max_length=config.max_model_len,
            max_width=config.max_span_width,
            map_location=config.device,
            dtype=self.torch_dtype,
        )
        self.model.eval()

        if config.quantization:
            logger.info("Applying quantization: %s", config.quantization)
            self.model.quantize(config.quantization)

        if torch.cuda.is_available():
            self.memory_estimator.measure_model_memory()

        self._supports_relations = self._detect_relation_support()
        logger.info("Relation extraction support: %s", self._supports_relations)

        self.collator = self.model.create_collator()

        if config.enable_sequence_packing:
            self.packing_config = InferencePackingConfig(
                max_length=config.max_model_len,
            )
            logger.info("Sequence packing enabled")
        else:
            self.packing_config = None

        if config.enable_compilation:
            self._precompile()

        if torch.cuda.is_available():
            self._calibrate_memory()

    def _detect_relation_support(self) -> bool:
        """Detect if the model supports relation extraction."""
        model_type = getattr(self.model.config, "model_type", "")
        return "relex" in model_type.lower()

    def _precompile(self) -> None:
        """Precompile model for configured batch sizes."""
        logger.info("Precompiling model for batch sizes: %s", self.config.precompiled_batch_sizes)

        self.model.compile()

        dummy_labels = ["person", "organization", "location"]
        dummy_relations = ["works_at", "located_in"] if self._supports_relations else None

        for batch_size in self.config.precompiled_batch_sizes:
            dummy_texts = [f"Sample text number {i} for precompilation warmup." for i in range(batch_size)]

            for _ in range(self.config.warmup_iterations):
                if self._supports_relations and dummy_relations:
                    self._run_batch_internal(
                        dummy_texts,
                        dummy_labels,
                        relations=dummy_relations,
                        threshold=0.5,
                        relation_threshold=0.5,
                        flat_ner=True,
                        multi_label=False,
                    )
                else:
                    self._run_batch_internal(
                        dummy_texts,
                        dummy_labels,
                        threshold=0.5,
                        flat_ner=True,
                        multi_label=False,
                    )

            logger.info("  Batch size %d: compiled", batch_size)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info("Precompilation complete.")

    def _calibrate_memory(self) -> None:
        """Build the memory calibration table across power-of-two seq lengths."""
        logger.info("Calibrating memory table...")

        self.memory_estimator.calibrate(
            self._run_batch_internal,
            max_seq_len=self.config.max_model_len,
            min_seq_len=self.config.calibration_min_seq_len,
        )

        logger.info("Memory calibration complete.")

    def batch_size_fn(self, seq_len: Optional[int] = None) -> int:
        """Largest precompiled batch size that fits at ``seq_len``.

        With no arguments, returns the worst-case answer (``max_model_len``),
        suitable for the deployment's initial ``max_batch_size``. Called again
        from ``_infer_batch`` with the observed seq length (text + label +
        relation words) to re-size Ray's batcher for the next accumulation.
        """
        if not torch.cuda.is_available():
            return self.config.precompiled_batch_sizes[-1]

        if seq_len is None:
            seq_len = self.config.max_model_len

        return self.memory_estimator.batch_size_fn(
            seq_len=seq_len,
            precompiled_sizes=self.config.precompiled_batch_sizes,
        )

    def observed_seq_len(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        relations: Optional[List[str]] = None,
    ) -> int:
        """Total input word count: longest text + all label/relation words.

        Labels and relations are concatenated into the input by the model, so
        they extend the effective sequence length for every sample in the
        batch.
        """
        max_text_words = max((len(t.split()) for t in texts if t.strip()), default=0)
        prompt_words = 0
        if labels:
            prompt_words += sum(len(label.split()) for label in labels)
        if relations:
            prompt_words += sum(len(r.split()) for r in relations)
        total = max_text_words + prompt_words
        return min(max(total, self.config.calibration_min_seq_len), self.config.max_model_len)

    def _filter_labels(self, labels: List[str]) -> List[str]:
        """Filter labels based on max_labels config."""
        if self.config.max_labels > 0 and len(labels) > self.config.max_labels:
            logger.warning("Truncating labels from %d to %d", len(labels), self.config.max_labels)
            return labels[: self.config.max_labels]
        return labels

    @torch.inference_mode()
    def _run_batch_internal(
        self,
        texts: List[str],
        labels: List[str],
        relations: Optional[List[str]] = None,
        threshold: float = 0.5,
        relation_threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> Union[List[List[Dict[str, Any]]], Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]]:
        """Run batch inference using low-level methods (no DataLoader).

        This is the core inference method that avoids DataLoader initialization
        overhead by directly using prepare_batch, collate_batch, run_batch,
        decode_batch, and map_entities_to_text.

        Args:
            texts: List of input texts.
            labels: Entity type labels.
            relations: Relation type labels (for relex models).
            threshold: Entity confidence threshold.
            relation_threshold: Relation confidence threshold.
            flat_ner: Whether to use flat NER.
            multi_label: Whether to allow multiple labels per span.

        Returns:
            For NER models: List of entity lists.
            For relex models: Tuple of (entities, relations) lists.
        """
        if self._supports_relations:
            return self._run_batch_relex(texts, labels, relations, threshold, relation_threshold, flat_ner, multi_label)
        else:
            return self._run_batch_ner(texts, labels, threshold, flat_ner, multi_label)

    def _run_batch_ner(
        self,
        texts: List[str],
        labels: List[str],
        threshold: float,
        flat_ner: bool,
        multi_label: bool,
    ) -> List[List[Dict[str, Any]]]:
        """Run NER batch inference using low-level methods."""
        prepared = self.model.prepare_batch(texts, labels)

        if not prepared["valid_texts"]:
            return [[] for _ in range(prepared["num_original"])]

        batch = self.model.collate_batch(
            prepared["input_x"],
            prepared["entity_types"],
            self.collator,
        )

        model_output = self.model.run_batch(
            batch,
            threshold=threshold,
            packing_config=self.packing_config,
            move_to_device=True,
        )

        decoded = self.model.decode_batch(
            model_output,
            batch,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )

        entity_results = self.model.map_entities_to_text(
            decoded,
            prepared["valid_texts"],
            prepared["valid_to_orig_idx"],
            prepared["start_token_map"],
            prepared["end_token_map"],
            prepared["num_original"],
        )

        return entity_results

    def _run_batch_relex(
        self,
        texts: List[str],
        labels: List[str],
        relations: Optional[List[str]],
        threshold: float,
        relation_threshold: float,
        flat_ner: bool,
        multi_label: bool,
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
        """Run relation extraction batch inference using low-level methods."""
        prepared = self.model.prepare_batch(texts, labels, relations=relations)

        if not prepared["valid_texts"]:
            num_orig = prepared["num_original"]
            return [[] for _ in range(num_orig)], [[] for _ in range(num_orig)]

        batch = self.model.collate_batch(
            prepared["input_x"],
            prepared["entity_types"],
            self.collator,
            relation_types=prepared.get("relation_types", []),
        )

        model_output = self.model.run_batch(
            batch,
            threshold=threshold,
            packing_config=self.packing_config,
            move_to_device=True,
        )

        decoded_entities, decoded_relations = self.model.decode_batch(
            model_output,
            batch,
            threshold=threshold,
            relation_threshold=relation_threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )

        entity_results = self.model.map_entities_to_text(
            decoded_entities,
            prepared["valid_texts"],
            prepared["valid_to_orig_idx"],
            prepared["start_token_map"],
            prepared["end_token_map"],
            prepared["num_original"],
        )

        relation_results = self.model.map_relations_to_text(
            decoded_relations,
            decoded_entities,
            prepared["valid_texts"],
            prepared["valid_to_orig_idx"],
            prepared["start_token_map"],
            prepared["end_token_map"],
            prepared["num_original"],
        )

        return entity_results, relation_results

    def predict(
        self,
        texts: Union[str, List[str]],
        labels: List[str],
        relations: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        relation_threshold: Optional[float] = None,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> List[Dict[str, Any]]:
        """Predict entities and optionally relations.

        Args:
            texts: Input text(s) to process.
            labels: Entity type labels to extract.
            relations: Relation type labels (only for relex models).
            threshold: Confidence threshold for entities.
            relation_threshold: Confidence threshold for relations.
            flat_ner: Whether to use flat NER (no overlapping entities).
            multi_label: Whether to allow multiple labels per span.

        Returns:
            List of result dicts, one per input text. Each dict contains:
                - "entities": List of entity dicts with start, end, text, label, score
                - "relations": List of relation dicts (only if model supports relations)
        """
        if isinstance(texts, str):
            texts = [texts]

        if threshold is None:
            threshold = self.config.default_threshold
        if relation_threshold is None:
            relation_threshold = self.config.default_relation_threshold

        labels = self._filter_labels(labels)

        if self._supports_relations and relations:
            entities, rels = self._run_batch_internal(
                texts,
                labels,
                relations=relations,
                threshold=threshold,
                relation_threshold=relation_threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )
            results = [{"entities": ents, "relations": r} for ents, r in zip(entities, rels)]
        else:
            entities = self._run_batch_internal(
                texts,
                labels,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )
            results = [{"entities": ents} for ents in entities]

        return results


def _build_deployment(config: GLiNERServeConfig):
    """Build Ray Serve deployment from config."""
    from ray import serve  # noqa: PLC0415

    batch_wait_s = max(config.batch_wait_timeout_ms, 0.0) / 1000.0
    initial_max_batch_size = config.max_batch_size

    @serve.deployment(
        num_replicas=config.num_replicas,
        ray_actor_options={
            "num_gpus": config.num_gpus_per_replica,
            "num_cpus": config.num_cpus_per_replica,
        },
        max_ongoing_requests=config.max_ongoing_requests,
    )
    class GLiNERDeployment:
        def __init__(self, serve_config: GLiNERServeConfig):
            self.server = GLiNERServer(serve_config)
            # Seed Ray's batcher with the pessimistic worst-case size so the
            # first batch is safe. ``_infer_batch`` re-calls ``batch_size_fn``
            # on every dispatch to re-size the batcher based on observed
            # sequence lengths.
            self._infer_batch.set_max_batch_size(self.server.batch_size_fn())
            logger.info(
                "Ray Serve batch size initialized to %d (precompiled: %s)",
                self.server.batch_size_fn(),
                serve_config.precompiled_batch_sizes,
            )

        @serve.batch(
            max_batch_size=initial_max_batch_size,
            batch_wait_timeout_s=batch_wait_s,
        )
        async def _infer_batch(
            self,
            texts: List[str],
            labels_list: List[List[str]],
            relations_list: List[Optional[List[str]]],
            thresholds: List[float],
            relation_thresholds: List[float],
            flat_ner_list: List[bool],
            multi_label_list: List[bool],
        ) -> List[Dict[str, Any]]:
            """Single forward pass over the Ray-accumulated batch.

            Before dispatch, re-sizes Ray's batcher via ``set_max_batch_size``
            using ``batch_size_fn`` on the observed seq length — so the next
            accumulation picks the largest precompiled size that fits.

            Assumes batch requests are homogeneous — labels/thresholds/flags
            are taken from the first request.
            """
            next_max_batch = self.server.batch_size_fn(
                seq_len=self.server.observed_seq_len(
                    texts,
                    labels=labels_list[0] if labels_list else None,
                    relations=relations_list[0] if relations_list else None,
                )
            )
            self._infer_batch.set_max_batch_size(next_max_batch)

            return self.server.predict(
                texts,
                labels_list[0],
                relations=relations_list[0],
                threshold=thresholds[0],
                relation_threshold=relation_thresholds[0],
                flat_ner=flat_ner_list[0],
                multi_label=multi_label_list[0],
            )

        async def predict(
            self,
            text: str,
            labels: List[str],
            relations: Optional[List[str]] = None,
            threshold: Optional[float] = None,
            relation_threshold: Optional[float] = None,
            flat_ner: bool = True,
            multi_label: bool = False,
        ) -> Dict[str, Any]:
            """Single prediction endpoint."""
            if threshold is None:
                threshold = self.server.config.default_threshold
            if relation_threshold is None:
                relation_threshold = self.server.config.default_relation_threshold

            results = await self._infer_batch(
                text,
                labels,
                relations,
                threshold,
                relation_threshold,
                flat_ner,
                multi_label,
            )
            return results

        async def __call__(self, request) -> Dict[str, Any]:
            """Handle HTTP requests."""
            payload = await request.json()
            return await self.predict(
                text=payload["text"],
                labels=payload["labels"],
                relations=payload.get("relations"),
                threshold=payload.get("threshold"),
                relation_threshold=payload.get("relation_threshold"),
                flat_ner=payload.get("flat_ner", True),
                multi_label=payload.get("multi_label", False),
            )

    return GLiNERDeployment.bind(config)


def serve(
    config: GLiNERServeConfig,
    blocking: bool = False,
) -> Any:
    """Start GLiNER Ray Serve deployment.

    Args:
        config: Server configuration.
        blocking: If True, blocks until the server is shut down.

    Returns:
        Ray Serve deployment handle for making predictions.

    Example:
        >>> from gliner.serve import GLiNERServeConfig, serve
        >>> config = GLiNERServeConfig(model="urchade/gliner_small-v2.1")
        >>> handle = serve(config)
        >>> # Make predictions
        >>> ref = handle.predict.remote("John works at Google", ["person", "org"])
        >>> print(ref.result())
    """
    import ray  # noqa: PLC0415
    from ray import serve as ray_serve  # noqa: PLC0415

    if not ray.is_initialized():
        ray.init(address=config.ray_address, ignore_reinit_error=True)

    ray_serve.start(detached=True)

    app = _build_deployment(config)
    handle = ray_serve.run(app, name="gliner", route_prefix=config.route_prefix)

    logger.info("GLiNER server running at http://localhost:8000%s", config.route_prefix)

    if blocking:
        import time  # noqa: PLC0415
        import signal  # noqa: PLC0415

        shutdown_event = False

        def handle_signal(_signum, _frame):
            nonlocal shutdown_event
            shutdown_event = True

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while not shutdown_event:
            time.sleep(1)

        ray_serve.shutdown()

    return handle


def shutdown() -> None:
    """Shutdown the GLiNER Ray Serve deployment."""
    from ray import serve as ray_serve  # noqa: PLC0415

    ray_serve.shutdown()


class GLiNERFactory:
    """vLLM-style synchronous facade over a GLiNER Ray Serve deployment.

    Bundles config → deploy → client into one lifecycle-managed object so
    callers never see Ray's ObjectRefs.

    Pass a list of texts to ``predict`` to preserve dynamic batching: each
    text is dispatched as a separate request so Ray Serve's ``@serve.batch``
    can accumulate them into a single forward pass. A Python loop of
    single-text calls would serialize and defeat batching.

    Example:
        >>> from gliner.serve import GLiNERFactory
        >>> llm = GLiNERFactory(model="urchade/gliner_small-v2.1")
        >>> outputs = llm.predict(
        ...     ["John works at Google", "Paris is in France"],
        ...     labels=["person", "organization", "location"],
        ... )
        >>> llm.shutdown()

        Or as a context manager:
        >>> with GLiNERFactory(model="urchade/gliner_small-v2.1") as llm:
        ...     out = llm.predict("John works at Google", ["person", "org"])
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        config: Optional[GLiNERServeConfig] = None,
        **kwargs,
    ):
        """Build a config (if not provided) and start the Ray Serve deployment.

        Args:
            model: Model name or path. Ignored if ``config`` is provided.
            config: Prebuilt ``GLiNERServeConfig``. Mutually exclusive with
                ``model``/``kwargs``.
            **kwargs: Forwarded to ``GLiNERServeConfig`` when building one.
        """
        if config is not None:
            if model is not None or kwargs:
                raise ValueError("Pass either `config` or `model`/kwargs, not both.")
        else:
            if model is None:
                raise ValueError("Must provide either `model` or `config`.")
            config = GLiNERServeConfig(model=model, **kwargs)

        self.config = config
        self._handle = serve(config, blocking=False)
        self._closed = False

    @property
    def handle(self):
        """Underlying Ray Serve deployment handle — for async/advanced use."""
        return self._handle

    def predict(
        self,
        texts: Union[str, List[str]],
        labels: List[str],
        relations: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        relation_threshold: Optional[float] = None,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Blocking prediction. Returns a dict for ``str`` input, list for list input."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)

        refs = [
            self._handle.predict.remote(
                t,
                labels,
                relations,
                threshold,
                relation_threshold,
                flat_ner,
                multi_label,
            )
            for t in items
        ]
        results = [ref.result() for ref in refs]
        return results[0] if single else results

    async def predict_async(
        self,
        texts: Union[str, List[str]],
        labels: List[str],
        relations: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        relation_threshold: Optional[float] = None,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Async prediction. Concurrent calls accumulate into one batch."""
        import asyncio  # noqa: PLC0415

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)

        refs = [
            self._handle.predict.remote(
                t,
                labels,
                relations,
                threshold,
                relation_threshold,
                flat_ner,
                multi_label,
            )
            for t in items
        ]
        results = list(await asyncio.gather(*refs))
        return results[0] if single else results

    def shutdown(self) -> None:
        """Tear down the Ray Serve deployment. Idempotent."""
        if self._closed:
            return
        from ray import serve as ray_serve  # noqa: PLC0415

        ray_serve.shutdown()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
