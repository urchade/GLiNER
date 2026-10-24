"""Ray Serve deployment for GLiNER with dynamic batching and memory-aware batch sizing."""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Any, Dict, List, Tuple

import torch

from gliner import GLiNER, InferencePackingConfig

from .api import APIError, AnalyzeEngine, capabilities_from_config
from .stats import BatchStats
from .config import GLiNERServeConfig
from .memory import GLiNERMemoryEstimator
from .api.errors import is_error_payload

logger = logging.getLogger(__name__)

LabelSet = List[str] | Dict[str, str]
Labels = LabelSet | List[LabelSet]


# One-word text used to pad batches up to a precompiled batch size.
_FILLER_TEXT = "filler"


def _min_batch_value(value):
    if isinstance(value, list):
        return min(value) if value else 0.5
    return value


def _normalize_relation_lists(relations):
    if relations is None:
        return None
    if isinstance(relations, list) and (not relations or isinstance(relations[0], (list, type(None)))):
        normalized = [item or [] for item in relations]
        return None if all(not item for item in normalized) else normalized
    return relations


def _labels_for_texts(labels: Labels, num_texts: int) -> List[LabelSet]:
    """Return one label set per text while preserving shared label sets."""
    if isinstance(labels, list) and labels and isinstance(labels[0], (list, dict)):
        if len(labels) != num_texts:
            raise ValueError(f"Per-text labels must have length {num_texts}, got {len(labels)}")
        return labels
    return [labels] * num_texts


def _label_word_count(labels: LabelSet) -> int:
    prompts = labels.values() if isinstance(labels, dict) else labels
    return sum(len(label.split()) for label in prompts)


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
        self.config = config
        self._polylora_model = None
        self._polylora_mixes_adapters = True
        self._adapter_id_re = re.compile(config.polylora_adapter_id_pattern)

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
        if config.enable_polylora:
            self._initialize_polylora()
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

        self.batch_stats.reset()  # count request traffic only, not warmup and calibration

    @property
    def batch_stats(self) -> BatchStats:
        """Forward passes by batch size since start-up (or the last reset)."""
        stats = self.__dict__.get("_batch_stats")
        if stats is None:
            stats = self.__dict__["_batch_stats"] = BatchStats()
        return stats

    def _initialize_polylora(self) -> None:
        try:
            from polylora import PolyLoraModel, PolyLoraConfig  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("enable_polylora=True requires the polylora package to be importable") from exc

        target_model = self._get_polylora_target_model()
        polylora_config = PolyLoraConfig(
            max_gpu_adapters=self.config.polylora_max_gpu_adapters,
            max_cpu_adapters=self.config.polylora_max_cpu_adapters,
            disk_cache_dir=self.config.polylora_disk_cache_dir,
            max_disk_adapters=self.config.polylora_max_disk_adapters,
            max_rank=self.config.polylora_max_rank,
            target_modules=self.config.polylora_adapter_weight_modules,
            base_adapter_id=self.config.polylora_base_adapter_id,
            use_triton_kernels=self.config.polylora_use_triton_kernels,
        )
        self._polylora_model = PolyLoraModel(target_model, polylora_config)
        self._set_polylora_target_model(self._polylora_model)
        # Stock HF DeBERTa projects its relative position embeddings once for the whole batch,
        # so one forward can carry only one adapter on those layers. FlashDeBERTa projects them
        # once per adapter (MultiLoraLinear.project_shared). Other encoders share no inputs.
        self._polylora_mixes_adapters = not any(
            type(module).__name__ == "DisentangledSelfAttention" for module in target_model.modules()
        )
        if not self._polylora_mixes_adapters:
            logger.info("PolyLoRA on HF DeBERTa: batches with mixed adapters run one adapter at a time")

        logger.info(
            "PolyLoRA enabled with %d GPU slots, max rank %d, disk cache %s",
            self.config.polylora_max_gpu_adapters,
            self.config.polylora_max_rank,
            self.config.polylora_disk_cache_dir or "disabled",
        )

    def _get_polylora_target_model(self):
        try:
            return self.model.model.token_rep_layer.bert_layer.model
        except AttributeError as exc:
            raise NotImplementedError("PolyLoRA is only implemented for GLiNER text encoder models") from exc

    def _set_polylora_target_model(self, wrapped_model) -> None:
        try:
            self.model.model.token_rep_layer.bert_layer.model = wrapped_model
        except AttributeError as exc:
            raise NotImplementedError("PolyLoRA is only implemented for GLiNER text encoder models") from exc

    def _validate_adapter_id(self, adapter_id: str) -> None:
        if not isinstance(adapter_id, str) or not self._adapter_id_re.fullmatch(adapter_id):
            raise ValueError("adapter_id must match polylora_adapter_id_pattern")
        if adapter_id == self.config.polylora_base_adapter_id:
            raise ValueError(f"{adapter_id!r} is reserved for base-only inference")

    def adapter_cache_status(self, adapter_id: str | None = None) -> Dict[str, Any]:
        if not self.config.enable_polylora or self._polylora_model is None:
            return {"enabled": False, "base_adapter_id": self.config.polylora_base_adapter_id}
        store = self._polylora_model.adapter_store
        disk_cache = getattr(store, "disk_cache", None)
        response: Dict[str, Any] = {
            "enabled": True,
            "base_adapter_id": self.config.polylora_base_adapter_id,
            "loaded": sorted(store.adapters.keys()),
            "disk_cached": sorted(disk_cache.entries.keys()) if disk_cache is not None else [],
            "disk_cache_dir": str(disk_cache.cache_dir) if disk_cache is not None else None,
            "max_disk_adapters": disk_cache.max_adapters if disk_cache is not None else None,
            "gpu_slots": list(self._polylora_model.adapter_cache.slot_to_adapter),
        }
        if adapter_id is not None:
            if adapter_id == self.config.polylora_base_adapter_id:
                response["adapter_id"] = adapter_id
                response["cached"] = True
                response["cpu_resident"] = False
                response["gpu_resident"] = True
                return response
            self._validate_adapter_id(adapter_id)
            response["adapter_id"] = adapter_id
            response["cached"] = disk_cache is not None and adapter_id in disk_cache
            response["cpu_resident"] = adapter_id in store.adapters
            response["gpu_resident"] = adapter_id in self._polylora_model.adapter_cache.adapter_to_slot
        return response

    def ensure_adapter_loaded(self, adapter_id: str | None) -> str | None:
        if adapter_id is None:
            return self.config.polylora_base_adapter_id if self.config.enable_polylora else None
        if adapter_id == self.config.polylora_base_adapter_id:
            return adapter_id if self.config.enable_polylora else None
        if not self.config.enable_polylora or self._polylora_model is None:
            raise KeyError(f"Unknown LoRA adapter id: {adapter_id}")
        self._validate_adapter_id(adapter_id)
        if adapter_id in self._polylora_model.adapter_store:
            return adapter_id
        raise KeyError(f"Unknown LoRA adapter id: {adapter_id}")

    def _resolve_adapter_ids(
        self,
        adapter_ids: str | List[str | None] | None,
        valid_to_orig_idx: List[int],
    ) -> str | List[str | None] | None:
        if not self.config.enable_polylora:
            if isinstance(adapter_ids, list):
                for adapter_id in adapter_ids:
                    if adapter_id not in (None, self.config.polylora_base_adapter_id):
                        self.ensure_adapter_loaded(adapter_id)
            elif adapter_ids not in (None, self.config.polylora_base_adapter_id):
                self.ensure_adapter_loaded(adapter_ids)
            return None
        if isinstance(adapter_ids, list):
            resolved = [self.ensure_adapter_loaded(adapter_ids[idx]) for idx in valid_to_orig_idx]
            return resolved
        # PolyLoRA takes one adapter id per batch row.
        adapter_id = self.ensure_adapter_loaded(adapter_ids)
        return [adapter_id for _ in valid_to_orig_idx]

    def _adapter_run_kwargs(self, adapter_ids: List[str | None] | None) -> Dict[str, Any]:
        """Model kwargs selecting one adapter per batch row.

        With PolyLoRA the ids are resolved to GPU cache slots here, right before the
        forward pass, and passed as the ``weight_indices`` tensor. Passing the id strings
        into a compiled model would make torch.compile guard on them and recompile for
        new adapter mixes.
        """
        if adapter_ids is None:
            return {}
        polylora_model = getattr(self, "_polylora_model", None)
        if polylora_model is None:
            return {"adapter_ids": adapter_ids}
        return {"weight_indices": polylora_model.resolve_adapters(adapter_ids)}

    def _detect_relation_support(self) -> bool:
        """Detect if the model supports relation extraction."""
        model_type = getattr(self.model.config, "model_type", "")
        return "relex" in model_type.lower()

    def _precompile(self) -> None:
        """Precompile model for configured batch sizes."""
        logger.info("Precompiling model for batch sizes: %s", self.config.precompiled_batch_sizes)

        # Each batch size is its own compiled entry (per-row Python lists such as token
        # lengths are specialised on their length). Past the limit torch silently runs
        # uncompiled, so leave room for every precompiled size.
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit, 2 * len(self.config.precompiled_batch_sizes)
        )
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

    def batch_size_fn(self, seq_len: int | None = None) -> int:
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
        labels: Labels | None = None,
        relations: List[str] | List[List[str] | None] | None = None,
    ) -> int:
        """Total input word count: longest text + all label/relation words.

        Labels and relations are concatenated into the input by the model, so
        they extend the effective sequence length for every sample in the
        batch.
        """
        max_text_words = max((len(t.split()) for t in texts if t.strip()), default=0)
        prompt_words = 0
        if labels:
            if isinstance(labels, list) and labels and isinstance(labels[0], (list, dict)):
                prompt_words += max(_label_word_count(label_set) for label_set in labels)
            else:
                prompt_words += _label_word_count(labels)
        if relations:
            normalized_relations = _normalize_relation_lists(relations)
            if normalized_relations:
                if isinstance(normalized_relations[0], list):
                    prompt_words += max(
                        sum(len(relation.split()) for relation in relation_set) for relation_set in normalized_relations
                    )
                else:
                    prompt_words += sum(len(r.split()) for r in normalized_relations)
        total = max_text_words + prompt_words
        return min(max(total, self.config.calibration_min_seq_len), self.config.max_model_len)

    def _filter_labels(self, labels: Labels) -> Labels:
        """Filter labels based on max_labels config."""
        if isinstance(labels, list) and labels and isinstance(labels[0], (list, dict)):
            return [self._filter_labels(label_set) for label_set in labels]
        if self.config.max_labels > 0 and len(labels) > self.config.max_labels:
            logger.warning("Truncating labels from %d to %d", len(labels), self.config.max_labels)
            if isinstance(labels, dict):
                return dict(list(labels.items())[: self.config.max_labels])
            return labels[: self.config.max_labels]
        return labels

    @torch.inference_mode()
    def _run_batch_internal(
        self,
        texts: List[str],
        labels: Labels,
        relations: List[str] | List[List[str] | None] | None = None,
        threshold: float | List[float] = 0.5,
        relation_threshold: float | List[float] = 0.5,
        flat_ner: bool | List[bool] = True,
        multi_label: bool | List[bool] = False,
        adapter_ids: str | List[str | None] | None = None,
    ) -> List[List[Dict[str, Any]]] | Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
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
            adapter_ids: Optional PolyLoRA adapter id or per-text adapter ids.

        Returns:
            For NER models: List of entity lists.
            For relex models: Tuple of (entities, relations) lists.
        """
        per_text_labels = isinstance(labels, list) and bool(labels) and isinstance(labels[0], (list, dict))
        if getattr(self.model, "is_runtime_model", False) and len(texts) > 1 and per_text_labels:
            if len(labels) != len(texts):
                raise ValueError(f"Per-text labels must have length {len(texts)}, got {len(labels)}")

            entity_results = []
            relation_results = []
            per_text_relations = bool(relations) and isinstance(relations[0], (list, type(None)))
            for index, text in enumerate(texts):
                result = self._run_batch_internal(
                    [text],
                    labels[index],
                    relations=relations[index] if per_text_relations else relations,
                    threshold=threshold[index] if isinstance(threshold, list) else threshold,
                    relation_threshold=(
                        relation_threshold[index] if isinstance(relation_threshold, list) else relation_threshold
                    ),
                    flat_ner=flat_ner[index] if isinstance(flat_ner, list) else flat_ner,
                    multi_label=multi_label[index] if isinstance(multi_label, list) else multi_label,
                    adapter_ids=adapter_ids[index] if isinstance(adapter_ids, list) else adapter_ids,
                )
                if self._supports_relations:
                    entities, found_relations = result
                    entity_results.extend(entities)
                    relation_results.extend(found_relations)
                else:
                    entity_results.extend(result)

            if self._supports_relations:
                return entity_results, relation_results
            return entity_results

        num_texts = len(texts)
        if (
            not getattr(self, "_polylora_mixes_adapters", True)
            and isinstance(adapter_ids, list)
            and len(set(adapter_ids)) > 1
        ):
            return self._run_batch_per_adapter(
                texts, labels, relations, threshold, relation_threshold, flat_ner, multi_label, adapter_ids
            )
        if not getattr(self.model, "is_runtime_model", False) and num_texts:
            (texts, labels, relations, threshold, relation_threshold, flat_ner, multi_label, adapter_ids) = (
                self._per_text_inputs(
                    texts, labels, relations, threshold, relation_threshold, flat_ner, multi_label, adapter_ids
                )
            )

        self.batch_stats.record(num_texts, len(texts))
        if self._supports_relations:
            entities, found_relations = self._run_batch_relex(
                texts,
                labels,
                relations,
                threshold,
                relation_threshold,
                flat_ner,
                multi_label,
                adapter_ids,
            )
            return entities[:num_texts], found_relations[:num_texts]
        return self._run_batch_ner(texts, labels, threshold, flat_ner, multi_label, adapter_ids)[:num_texts]

    def _run_batch_per_adapter(
        self, texts, labels, relations, threshold, relation_threshold, flat_ner, multi_label, adapter_ids
    ):
        """Run a mixed-adapter batch as one sub-batch per adapter, in the original order."""
        (_, labels, relations, threshold, relation_threshold, flat_ner, multi_label, adapter_ids) = (
            self._per_text_inputs(  # one value per text, no padding (compilation is checked below)
                texts,
                labels,
                relations,
                threshold,
                relation_threshold,
                flat_ner,
                multi_label,
                adapter_ids,
                pad=False,
            )
        )
        groups: Dict[Any, List[int]] = {}
        for index, adapter_id in enumerate(adapter_ids):
            groups.setdefault(adapter_id, []).append(index)

        entities: List[Any] = [None] * len(texts)
        found_relations: List[Any] = [None] * len(texts)
        for adapter_id, indices in groups.items():

            def pick(values, indices=indices):
                return None if values is None else [values[i] for i in indices]

            result = self._run_batch_internal(
                pick(texts),
                pick(labels),
                pick(relations),
                pick(threshold),
                pick(relation_threshold),
                pick(flat_ner),
                pick(multi_label),
                adapter_id,
            )
            group_entities, group_relations = result if self._supports_relations else (result, None)
            for position, index in enumerate(indices):
                entities[index] = group_entities[position]
                if group_relations is not None:
                    found_relations[index] = group_relations[position]
        if self._supports_relations:
            return entities, found_relations
        return entities

    def _compiled_batch_padding(self, num_texts: int) -> int:
        """Filler rows that round a batch up to the next precompiled batch size."""
        if not self.config.enable_compilation:
            return 0
        for size in self.config.precompiled_batch_sizes:
            if size >= num_texts:
                return size - num_texts
        return 0

    def _per_text_inputs(
        self, texts, labels, relations, threshold, relation_threshold, flat_ner, multi_label, adapter_ids, pad=True
    ) -> tuple:
        """Give every argument one value per text, and pad compiled batches.

        All callers (warmup, memory calibration, ``predict`` and Ray's batcher) then
        reach the compiled model with inputs of the same structure, so warmup compiles
        what requests use. With compilation on, the batch is padded with one-word filler
        texts up to the next precompiled batch size, so request batches of any size reuse
        the warmed-up graphs instead of recompiling. Callers drop the filler outputs.
        """
        num_texts = len(texts)

        def per_text(value):
            if isinstance(value, list):
                if len(value) != num_texts:
                    raise ValueError(f"Per-text values must have length {num_texts}, got {len(value)}")
                return list(value)
            return [value] * num_texts

        labels = list(_labels_for_texts(labels, num_texts))
        if relations is not None:
            relations = (
                per_text(relations)
                if relations and isinstance(relations[0], (list, type(None)))
                else [relations] * num_texts
            )
        threshold = per_text(threshold)
        relation_threshold = per_text(relation_threshold)
        flat_ner = per_text(flat_ner)
        multi_label = per_text(multi_label)
        adapter_ids = per_text(adapter_ids)

        padding = self._compiled_batch_padding(num_texts) if pad else 0
        if padding:
            texts = list(texts) + [_FILLER_TEXT] * padding
            labels += [labels[0]] * padding
            if relations is not None:
                relations += [relations[0]] * padding
            threshold += [threshold[0]] * padding
            relation_threshold += [relation_threshold[0]] * padding
            flat_ner += [flat_ner[0]] * padding
            multi_label += [multi_label[0]] * padding
            adapter_ids += [None] * padding  # filler rows run on the base model
        return texts, labels, relations, threshold, relation_threshold, flat_ner, multi_label, adapter_ids

    def _run_batch_ner(
        self,
        texts: List[str],
        labels: Labels,
        threshold: float | List[float],
        flat_ner: bool | List[bool],
        multi_label: bool | List[bool],
        adapter_ids: str | List[str | None] | None = None,
    ) -> List[List[Dict[str, Any]]]:
        """Run NER batch inference using low-level methods."""
        prepared = self.model.prepare_batch(texts, labels)

        if not prepared["valid_texts"]:
            return [[] for _ in range(prepared["num_original"])]

        resolved_adapter_ids = self._resolve_adapter_ids(adapter_ids, prepared["valid_to_orig_idx"])

        batch = self.model.collate_batch(
            prepared["input_x"],
            prepared["entity_types"],
            self.collator,
            label_names=prepared["label_names"],
        )

        run_kwargs = self._adapter_run_kwargs(resolved_adapter_ids)

        model_output = self.model.run_batch(
            batch,
            threshold=_min_batch_value(threshold),
            packing_config=self.packing_config,
            move_to_device=True,
            **run_kwargs,
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
        labels: Labels,
        relations: List[str] | List[List[str] | None] | None,
        threshold: float | List[float],
        relation_threshold: float | List[float],
        flat_ner: bool | List[bool],
        multi_label: bool | List[bool],
        adapter_ids: str | List[str | None] | None = None,
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
        """Run relation extraction batch inference using low-level methods."""
        relations = _normalize_relation_lists(relations)
        prepared = self.model.prepare_batch(texts, labels, relations=relations)

        if not prepared["valid_texts"]:
            num_orig = prepared["num_original"]
            return [[] for _ in range(num_orig)], [[] for _ in range(num_orig)]

        resolved_adapter_ids = self._resolve_adapter_ids(adapter_ids, prepared["valid_to_orig_idx"])

        batch = self.model.collate_batch(
            prepared["input_x"],
            prepared["entity_types"],
            self.collator,
            relation_types=prepared.get("relation_types", []),
            label_names=prepared["label_names"],
        )

        run_kwargs = self._adapter_run_kwargs(resolved_adapter_ids)

        model_output = self.model.run_batch(
            batch,
            threshold=_min_batch_value(threshold),
            adjacency_threshold=_min_batch_value(relation_threshold),
            packing_config=self.packing_config,
            move_to_device=True,
            **run_kwargs,
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
        texts: str | List[str],
        labels: Labels,
        relations: List[str] | None = None,
        threshold: float | None = None,
        relation_threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        adapter_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Predict entities and optionally relations.

        Args:
            texts: Input text(s) to process.
            labels: Entity labels shared by all texts, or one label set per text.
            relations: Relation type labels (only for relex models).
            threshold: Confidence threshold for entities.
            relation_threshold: Confidence threshold for relations.
            flat_ner: Whether to use flat NER (no overlapping entities).
            multi_label: Whether to allow multiple labels per span.
            adapter_id: Optional PolyLoRA adapter id to use for inference.

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

        return self.run_rows_batch(
            texts,
            labels,
            relations,
            threshold,
            relation_threshold,
            flat_ner,
            multi_label,
            adapter_id,
        )

    def run_rows_batch(
        self,
        texts: List[str],
        labels: Labels,
        relations: List[str] | List[List[str] | None] | None,
        threshold: float | List[float],
        relation_threshold: float | List[float],
        flat_ner: bool | List[bool],
        multi_label: bool | List[bool],
        adapter_ids: str | List[str | None] | None,
    ) -> List[Dict[str, Any]]:
        """Run one model batch and return ``{"entities", "relations"?}`` per text.

        Every argument is either shared by all texts or a list with one value
        per text. ``"relations"`` is present when relations were requested.
        """
        labels = self._filter_labels(labels)
        relations = _normalize_relation_lists(relations)

        if self._supports_relations and relations:
            entities, rels = self._run_batch_internal(
                texts,
                labels,
                relations=relations,
                threshold=threshold,
                relation_threshold=relation_threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
                adapter_ids=adapter_ids,
            )
            return [{"entities": ents, "relations": r} for ents, r in zip(entities, rels, strict=False)]

        result = self._run_batch_internal(
            texts,
            labels,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            adapter_ids=adapter_ids,
        )
        # Relation models return (entities, relations) even when no relations were asked for.
        entities = result[0] if self._supports_relations else result
        return [{"entities": ents} for ents in entities]

    # -- analyze API -------------------------------------------------------------

    @property
    def analyze_engine(self) -> AnalyzeEngine:
        """Request planner and result assembler of ``/v1/analyze``."""
        engine = getattr(self, "_analyze_engine", None)
        if engine is None:
            words_splitter = self.model.data_processor.words_splitter
            max_words = min(
                int(getattr(self.model.config, "max_len", self.config.max_model_len)), self.config.max_model_len
            )
            engine = self._analyze_engine = AnalyzeEngine(
                capabilities_from_config(self.config, supports_relations=self._supports_relations),
                max_words=max_words,
                split_words=lambda text: [(start, end) for _, start, end in words_splitter(text)],
                resolve_adapter=self.ensure_adapter_loaded,
                window_overlap=self.config.analyze_window_overlap,
            )
        return engine

    def run_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run analyze rows in memory-safe chunks (in-process, without Ray's batcher)."""
        outputs: List[Dict[str, Any]] = []
        chunk = max(1, self.config.max_batch_size)
        index = 0
        while index < len(rows):
            part = rows[index : index + chunk]
            fitting = self.batch_size_fn(
                seq_len=self.observed_seq_len(
                    [row["text"] for row in part],
                    labels=[row["labels"] for row in part],
                    relations=[row["relations"] for row in part],
                )
            )
            part = part[: max(1, fitting)]
            outputs.extend(
                self.run_rows_batch(
                    [row["text"] for row in part],
                    [row["labels"] for row in part],
                    [row["relations"] for row in part],
                    [row["threshold"] for row in part],
                    [row["relation_threshold"] for row in part],
                    [row["flat_ner"] for row in part],
                    [row["multi_label"] for row in part],
                    [row["adapter_id"] for row in part],
                )
            )
            index += len(part)
        return outputs

    def analyze(self, body: Any) -> Dict[str, Any]:
        """``POST /v1/analyze`` in-process. Raises ``APIError`` for a rejected request."""
        return self.analyze_engine.analyze(body, self.run_rows)

    def analyze_batch(self, body: Any) -> Dict[str, Any]:
        """``POST /v1/analyze/batch`` in-process. Raises ``APIError`` for a bad envelope."""
        return self.analyze_engine.analyze_batch(body, self.run_rows)

    def capabilities(self, model_id: str | None = None) -> Dict[str, Any]:
        info = self.analyze_engine.capabilities
        if model_id is not None and model_id not in info.model_ids():
            raise APIError("MODEL_NOT_FOUND", f"Model {model_id!r} is not served here", "/model")
        return info.to_dict()

    def models(self) -> Dict[str, Any]:
        info = self.analyze_engine.capabilities
        return {"data": [{"id": info.model, "aliases": list(info.aliases)}]}

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": self.analyze_engine.capabilities.model,
            "device": self.config.device,
            "cuda_available": torch.cuda.is_available(),
            "polylora": self.config.enable_polylora,
            "sequence_packing": self.packing_config is not None,
            "max_batch_size": self.batch_size_fn(),
            "task_types": list(self.analyze_engine.capabilities.task_types),
        }


async def _read_json_body(request, max_bytes: int) -> Any:
    """Parse a JSON request body, refusing bodies over ``max_bytes`` (``PAYLOAD_TOO_LARGE``)."""
    content_length = request.headers.get("content-length")
    if max_bytes > 0 and content_length and content_length.isdigit() and int(content_length) > max_bytes:
        raise APIError("PAYLOAD_TOO_LARGE", f"Request body exceeds {max_bytes} bytes")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if 0 < max_bytes < len(body):
            raise APIError("PAYLOAD_TOO_LARGE", f"Request body exceeds {max_bytes} bytes")
    try:
        return json.loads(bytes(body))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise APIError("INVALID_REQUEST", f"Request body is not valid JSON: {exc}", "") from exc


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
        max_queued_requests=config.queue_capacity,
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
            labels_list: List[LabelSet],
            relations_list: List[List[str] | None],
            thresholds: List[float],
            relation_thresholds: List[float],
            flat_ner_list: List[bool],
            multi_label_list: List[bool],
            adapter_ids: List[str | None],
        ) -> List[Dict[str, Any]]:
            """Single forward pass over the Ray-accumulated batch.

            Before dispatch, re-sizes Ray's batcher via ``set_max_batch_size``
            using ``batch_size_fn`` on the observed seq length — so the next
            accumulation picks the largest precompiled size that fits.

            Supports heterogeneous request parameters by passing per-text
            labels, relations, thresholds, and decode flags through to the
            model decode path.
            """
            next_max_batch = self.server.batch_size_fn(
                seq_len=self.server.observed_seq_len(
                    texts,
                    labels=labels_list,
                    relations=relations_list,
                )
            )
            self._infer_batch.set_max_batch_size(next_max_batch)

            return self.server.run_rows_batch(
                texts,
                labels_list,
                relations_list,
                thresholds,
                relation_thresholds,
                flat_ner_list,
                multi_label_list,
                adapter_ids,
            )

        async def _run_analyze_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
            """One analyze row through the dynamic batcher, shared with ``predict``."""
            return await self._infer_batch(
                row["text"],
                row["labels"],
                row["relations"],
                row["threshold"],
                row["relation_threshold"],
                row["flat_ner"],
                row["multi_label"],
                row["adapter_id"],
            )

        async def analyze(self, body: Any) -> Dict[str, Any]:
            """``/v1/analyze``: the response, or an ``{"error": ...}`` object."""
            engine = self.server.analyze_engine
            outcome = (await engine.analyze_many_async([body], self._run_analyze_row))[0]
            return outcome.to_dict() if isinstance(outcome, APIError) else outcome

        async def analyze_batch(self, body: Any) -> Dict[str, Any]:
            """``/v1/analyze/batch``: the response, or an ``{"error": ...}`` object."""
            try:
                return await self.server.analyze_engine.analyze_batch_async(body, self._run_analyze_row)
            except APIError as exc:
                return exc.to_dict()

        def capabilities(self, model_id: str | None = None) -> Dict[str, Any]:
            try:
                return self.server.capabilities(model_id)
            except APIError as exc:
                return exc.to_dict()

        def models(self) -> Dict[str, Any]:
            return self.server.models()

        def health(self) -> Dict[str, Any]:
            return self.server.health()

        def batch_stats(self, reset: bool = False) -> Dict[str, Any]:
            """Forward passes by batch size; ``reset`` starts a new count."""
            stats = self.server.batch_stats.to_dict()
            if reset:
                self.server.batch_stats.reset()
            return stats

        async def _handle_v1(self, request, path: str):  # noqa: PLR0911
            """Routes of the analyze API (``{route_prefix}/v1/...``)."""
            from starlette.responses import JSONResponse  # noqa: PLC0415

            def respond(payload: Dict[str, Any]) -> JSONResponse:
                status = APIError.from_dict(payload).status if is_error_payload(payload) else 200
                return JSONResponse(payload, status_code=status)

            route = path[path.find("/v1/") :]
            method = request.method
            if route in ("/v1/analyze", "/v1/analyze/batch"):
                if method != "POST":
                    return respond(APIError("INVALID_REQUEST", f"Use POST for {route}").to_dict())
                try:
                    body = await _read_json_body(request, self.server.config.max_request_bytes)
                except APIError as exc:
                    return respond(exc.to_dict())
                if route == "/v1/analyze":
                    return respond(await self.analyze(body))
                return respond(await self.analyze_batch(body))
            if method != "GET":
                return respond(APIError("INVALID_REQUEST", f"Use GET for {route}").to_dict())
            if route in ("/v1/health", "/v1/ready"):
                return respond(self.health())
            if route == "/v1/models":
                return respond(self.models())
            if route == "/v1/adapter-cache":
                try:
                    return respond(self.server.adapter_cache_status(request.query_params.get("adapter_id")))
                except ValueError as exc:
                    return respond(APIError("INVALID_REQUEST", str(exc), "/adapter_id").to_dict())
            match = re.fullmatch(r"/v1/models/(.+)/capabilities", route)
            if match:
                return respond(self.capabilities(match.group(1)))
            return JSONResponse(
                {"error": {"code": "INVALID_REQUEST", "message": f"Unknown route {route}"}}, status_code=404
            )

        async def predict(
            self,
            text: str,
            labels: LabelSet,
            relations: List[str] | None = None,
            threshold: float | None = None,
            relation_threshold: float | None = None,
            flat_ner: bool = True,
            multi_label: bool = False,
            adapter_id: str | None = None,
        ) -> Dict[str, Any]:
            """Single prediction endpoint."""
            if threshold is None:
                threshold = self.server.config.default_threshold
            if relation_threshold is None:
                relation_threshold = self.server.config.default_relation_threshold
            # Reject unknown adapters before batching: inside the batch, the error would fail
            # every request batched with this one.
            self.server.ensure_adapter_loaded(adapter_id)

            results = await self._infer_batch(
                text,
                labels,
                relations,
                threshold,
                relation_threshold,
                flat_ner,
                multi_label,
                adapter_id,
            )
            return results

        async def __call__(self, request) -> Dict[str, Any]:
            """Handle HTTP requests."""
            from starlette.responses import JSONResponse  # noqa: PLC0415

            path = request.url.path.rstrip("/")
            if "/v1/" in path:
                return await self._handle_v1(request, path)
            if path.endswith("/adapter-cache"):
                adapter_id = request.query_params.get("adapter_id")
                try:
                    return self.server.adapter_cache_status(adapter_id)
                except ValueError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=400)

            payload = await request.json()
            try:
                return await self.predict(
                    text=payload["text"],
                    labels=payload["labels"],
                    relations=payload.get("relations"),
                    threshold=payload.get("threshold"),
                    relation_threshold=payload.get("relation_threshold"),
                    flat_ner=payload.get("flat_ner", True),
                    multi_label=payload.get("multi_label", False),
                    adapter_id=payload.get("adapter_id"),
                )
            except KeyError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            except (TypeError, ValueError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)

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

    ray_serve.start(detached=True, http_options={"port": config.http_port})

    app = _build_deployment(config)
    handle = ray_serve.run(app, name="gliner", route_prefix=config.route_prefix)

    logger.info("GLiNER server running at http://localhost:%d%s", config.http_port, config.route_prefix)

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
        model: str | None = None,
        *,
        config: GLiNERServeConfig | None = None,
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
        texts: str | List[str],
        labels: Labels,
        relations: List[str] | None = None,
        threshold: float | None = None,
        relation_threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        adapter_id: str | None = None,
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Blocking prediction. Returns a dict for ``str`` input, list for list input."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        labels_list = _labels_for_texts(labels, len(items))

        refs = [
            self._handle.predict.remote(
                t,
                label_set,
                relations,
                threshold,
                relation_threshold,
                flat_ner,
                multi_label,
                adapter_id,
            )
            for t, label_set in zip(items, labels_list, strict=False)
        ]
        results = [ref.result() for ref in refs]
        return results[0] if single else results

    # -- analyze API ---------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.config.model_id or self.config.model

    def _analyze_body(self, input: Any, tasks: Dict[str, Any] | None, options, model, body) -> Dict[str, Any]:
        if body is not None:
            return body
        request: Dict[str, Any] = {"model": model or self.model_id, "input": input, "tasks": tasks}
        if options is not None:
            request["options"] = options
        return request

    def _batch_body(self, items: List[Dict[str, Any]], options, model) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model or self.model_id, "items": items}
        if options is not None:
            body["options"] = options
        return body

    @staticmethod
    def _unwrap(payload: Dict[str, Any]) -> Dict[str, Any]:
        if is_error_payload(payload):
            raise APIError.from_dict(payload)
        return payload

    def analyze(
        self,
        input: Any = None,
        tasks: Dict[str, Any] | None = None,
        *,
        options: Dict[str, Any] | None = None,
        model: str | None = None,
        body: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Run one ``/v1/analyze`` request; raises ``APIError`` when it is rejected.

        Pass ``input`` and ``tasks`` (``model`` defaults to the served model), or a
        complete request as ``body``.
        """
        request = self._analyze_body(input, tasks, options, model, body)
        return self._unwrap(self._handle.analyze.remote(request).result())

    def analyze_batch(
        self,
        items: List[Dict[str, Any]],
        *,
        options: Dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Dict[str, Any]:
        """Run a ``/v1/analyze/batch`` request; item errors are returned per item."""
        return self._unwrap(self._handle.analyze_batch.remote(self._batch_body(items, options, model)).result())

    async def analyze_async(
        self,
        input: Any = None,
        tasks: Dict[str, Any] | None = None,
        *,
        options: Dict[str, Any] | None = None,
        model: str | None = None,
        body: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Async ``analyze``. Concurrent calls share model batches."""
        request = self._analyze_body(input, tasks, options, model, body)
        return self._unwrap(await self._handle.analyze.remote(request))

    async def analyze_batch_async(
        self,
        items: List[Dict[str, Any]],
        *,
        options: Dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Dict[str, Any]:
        return self._unwrap(await self._handle.analyze_batch.remote(self._batch_body(items, options, model)))

    def capabilities(self, model_id: str | None = None) -> Dict[str, Any]:
        return self._unwrap(self._handle.capabilities.remote(model_id).result())

    async def predict_async(
        self,
        texts: str | List[str],
        labels: Labels,
        relations: List[str] | None = None,
        threshold: float | None = None,
        relation_threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        adapter_id: str | None = None,
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Async prediction. Concurrent calls accumulate into one batch."""
        import asyncio  # noqa: PLC0415

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        labels_list = _labels_for_texts(labels, len(items))

        refs = [
            self._handle.predict.remote(
                t,
                label_set,
                relations,
                threshold,
                relation_threshold,
                flat_ner,
                multi_label,
                adapter_id,
            )
            for t, label_set in zip(items, labels_list, strict=False)
        ]
        results = list(await asyncio.gather(*refs))
        return results[0] if single else results

    def shutdown(self) -> None:
        """Tear down the Ray Serve deployment and the Ray runtime it booted.

        Idempotent. Shutting down Ray after Serve avoids leaving the driver
        attached to a detached Serve instance — the latter produces noisy
        ``ServeController ... killed by ray.kill`` retry warnings in the
        raylet log when the process exits.
        """
        if self._closed:
            return
        import ray  # noqa: PLC0415
        from ray import serve as ray_serve  # noqa: PLC0415

        ray_serve.shutdown()
        if ray.is_initialized():
            ray.shutdown()
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
