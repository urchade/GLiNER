"""Session cache support for decoder-backed span inference."""

from __future__ import annotations

from typing import Any, Iterable, Sequence
from threading import RLock
from dataclasses import field, replace, dataclass

import torch

try:
    from transformers.cache_utils import DynamicCache, DynamicLayer
except ImportError:  # transformers < 5 uses list-backed DynamicCache storage.
    from transformers.cache_utils import DynamicCache

    DynamicLayer = None


DEFAULT_CACHE_INITIAL_CAPACITY = 256


def _empty_long() -> torch.Tensor:
    return torch.empty(0, dtype=torch.long)


def _device_matches(actual: torch.device, requested: torch.device) -> bool:
    return actual.type == requested.type and (
        requested.index is None or actual.index == requested.index
    )


if DynamicLayer is not None:

    class ReusableDynamicLayer(DynamicLayer):
        """Append-only KV storage that grows geometrically instead of every turn."""

        def __init__(
            self,
            initial_capacity: int = DEFAULT_CACHE_INITIAL_CAPACITY,
            max_cache_length: int | None = None,
        ) -> None:
            super().__init__()
            self.initial_capacity = initial_capacity
            self.maximum_length = max_cache_length
            self.current_length = 0
            self.capacity = 0

        def _next_capacity(self, required: int) -> int:
            if self.maximum_length is not None and required > self.maximum_length:
                raise ValueError(
                    f"KV cache requires {required} positions, exceeding its "
                    f"maximum length of {self.maximum_length}"
                )
            capacity = max(1, self.initial_capacity, self.capacity)
            while capacity < required:
                capacity *= 2
            if self.maximum_length is not None:
                capacity = min(capacity, self.maximum_length)
            return capacity

        @staticmethod
        def _allocate(source: torch.Tensor, capacity: int) -> torch.Tensor:
            shape = (*source.shape[:-2], capacity, source.shape[-1])
            return source.new_empty(shape)

        def lazy_initialization(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
        ) -> None:
            self.dtype, self.device = key_states.dtype, key_states.device
            self.capacity = self._next_capacity(key_states.shape[-2])
            self.keys = self._allocate(key_states, self.capacity)
            self.values = self._allocate(value_states, self.capacity)
            self.current_length = 0
            self.is_initialized = True

        def _grow(self, required: int) -> None:
            if required <= self.capacity:
                return
            new_capacity = self._next_capacity(required)
            new_keys = self._allocate(self.keys, new_capacity)
            new_values = self._allocate(self.values, new_capacity)
            if self.current_length:
                active = slice(0, self.current_length)
                new_keys[..., active, :].copy_(self.keys[..., active, :])
                new_values[..., active, :].copy_(self.values[..., active, :])
            self.keys = new_keys
            self.values = new_values
            self.capacity = new_capacity

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            *args,
            **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del args, kwargs
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)

            new_length = key_states.shape[-2]
            required = self.current_length + new_length
            self._grow(required)
            destination = slice(self.current_length, required)
            self.keys[..., destination, :].copy_(key_states)
            self.values[..., destination, :].copy_(value_states)
            self.current_length = required
            active = slice(0, required)
            return self.keys[..., active, :], self.values[..., active, :]

        def get_seq_length(self) -> int:
            return self.current_length if self.is_initialized else 0

        def get_max_cache_shape(self) -> int:
            # The active attention length remains dynamic even though backing
            # storage is reserved in larger blocks.
            return -1

        @property
        def max_batch_size(self) -> int:
            return self.keys.shape[0] if self.is_initialized else 0

        @property
        def max_cache_len(self) -> int:
            return self.maximum_length if self.maximum_length is not None else -1

        def reset(self) -> None:
            self.current_length = 0

        def crop(self, max_length: int) -> None:
            if max_length < 0:
                max_length = self.current_length - abs(max_length)
            self.current_length = max(0, min(self.current_length, max_length))

        def batch_repeat_interleave(self, repeats: int) -> None:
            if self.is_initialized:
                self.keys = self.keys.repeat_interleave(repeats, dim=0)
                self.values = self.values.repeat_interleave(repeats, dim=0)

        def batch_select_indices(self, indices: torch.Tensor) -> None:
            if self.is_initialized:
                self.keys = self.keys[indices, ...]
                self.values = self.values[indices, ...]

        def move_to(self, device: torch.device | str) -> None:
            if not self.is_initialized:
                return
            self.keys = self.keys.to(device)
            self.values = self.values.to(device)
            self.device = self.keys.device

else:
    ReusableDynamicLayer = None


class ReusableDynamicCache(DynamicCache):
    """DynamicCache with stable, capacity-backed tensors when supported."""

    def __init__(
        self,
        config=None,
        initial_capacity: int = DEFAULT_CACHE_INITIAL_CAPACITY,
        max_cache_length: int | None = None,
    ) -> None:
        if initial_capacity < 1:
            raise ValueError("initial_capacity must be positive")
        self.initial_capacity = initial_capacity
        self.maximum_length = max_cache_length

        if DynamicLayer is None:
            # transformers 4.x has a list-backed cache API. Keep its compatible
            # behavior; CacheState still avoids reconstructing it on every turn.
            super().__init__()
            self.uses_reusable_storage = False
            return

        super().__init__(config=config)
        self.uses_reusable_storage = True

        def new_layer():
            return ReusableDynamicLayer(initial_capacity, max_cache_length)

        for layer_index, layer in enumerate(self.layers):
            # Preserve bounded sliding-window and recurrent/hybrid cache layers.
            if type(layer) is DynamicLayer:
                self.layers[layer_index] = new_layer()
        if self.layer_class_to_replicate is DynamicLayer:
            self.layer_class_to_replicate = new_layer

    def move_to(self, device: torch.device | str):
        if hasattr(self, "layers"):
            for layer in self.layers:
                if isinstance(layer, ReusableDynamicLayer):
                    layer.move_to(device)
                elif getattr(layer, "is_initialized", False):
                    layer.keys = layer.keys.to(device)
                    layer.values = layer.values.to(device)
                    if hasattr(layer, "device"):
                        layer.device = layer.keys.device
        else:
            self.key_cache = [tensor.to(device) for tensor in self.key_cache]
            self.value_cache = [tensor.to(device) for tensor in self.value_cache]
        return self


def create_reusable_cache(
    config,
    *,
    initial_capacity: int = DEFAULT_CACHE_INITIAL_CAPACITY,
    max_cache_length: int | None = None,
) -> ReusableDynamicCache:
    """Create a session KV cache whose storage is reused between appends."""
    if max_cache_length is not None:
        initial_capacity = min(initial_capacity, max_cache_length)
    return ReusableDynamicCache(
        config=config,
        initial_capacity=initial_capacity,
        max_cache_length=max_cache_length,
    )


@dataclass
class CacheState:
    """All reusable state belonging to one streaming-span session."""

    past_key_values: Any = None
    input_ids: torch.Tensor = field(default_factory=_empty_long)
    attention_mask: torch.Tensor = field(default_factory=_empty_long)
    token_word_mask: torch.Tensor = field(default_factory=_empty_long)
    past_word_embeddings: torch.Tensor | None = None
    past_word_mask: torch.Tensor | None = None
    prompts_embedding: torch.Tensor | None = None
    prompts_mask: torch.Tensor | None = None
    cached_length: int = 0
    next_position_id: int | None = None
    session_id: str | None = None
    labels: tuple[str, ...] = ()
    text: str = ""
    tokens: list[str] = field(default_factory=list)
    char_starts: list[int] = field(default_factory=list)
    char_ends: list[int] = field(default_factory=list)
    span_logits: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def next_position(self) -> int:
        """Return the absolute decoder position for the next token."""
        return self.cached_length if self.next_position_id is None else self.next_position_id

    @property
    def word_length(self) -> int:
        """Return the authoritative cached word count without a device sync."""
        return len(self.tokens)

    def to(self, device: torch.device | str) -> CacheState:
        """Move reusable model tensors only when the target device changes."""
        requested = torch.device(device)
        model_tensors = (
            self.input_ids,
            self.attention_mask,
            self.token_word_mask,
            self.past_word_embeddings,
            self.past_word_mask,
            self.prompts_embedding,
            self.prompts_mask,
        )
        tensors_match = all(
            tensor is None or _device_matches(tensor.device, requested)
            for tensor in model_tensors
        )
        if tensors_match and _past_kv_on_device(self.past_key_values, requested):
            # The normal streaming path stays on one device. Returning the exact
            # state preserves the cache object and its preallocated KV storage.
            return self

        return replace(
            self,
            past_key_values=_move_past_kv(self.past_key_values, device),
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            token_word_mask=self.token_word_mask.to(device),
            past_word_embeddings=(
                self.past_word_embeddings.to(device) if self.past_word_embeddings is not None else None
            ),
            past_word_mask=self.past_word_mask.to(device) if self.past_word_mask is not None else None,
            prompts_embedding=self.prompts_embedding.to(device) if self.prompts_embedding is not None else None,
            prompts_mask=self.prompts_mask.to(device) if self.prompts_mask is not None else None,
            tokens=list(self.tokens),
            char_starts=list(self.char_starts),
            char_ends=list(self.char_ends),
            # Span scores intentionally remain on CPU so long-running sessions
            # do not retain an ever-growing GPU prediction history.
            span_logits=self.span_logits.copy(),
            metadata=self.metadata.copy(),
        )


class SessionCacheManager:
    """Thread-safe ownership and lifecycle management for session states."""

    def __init__(self) -> None:
        self._states: dict[str, CacheState] = {}
        self._lock = RLock()

    def get(self, session_id: str) -> CacheState | None:
        with self._lock:
            return self._states.get(session_id)

    def put(self, state: CacheState) -> None:
        if state.session_id is None:
            raise ValueError("A cached state must have a session_id")
        with self._lock:
            self._states[state.session_id] = state

    def clear(self, session_ids: str | Iterable[str] | None = None) -> None:
        with self._lock:
            if session_ids is None:
                self._states.clear()
                return
            if isinstance(session_ids, str):
                session_ids = [session_ids]
            for session_id in session_ids:
                self._states.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)


def _cache_layers(past_kv):
    """Yield ``(layer_index, keys, values)`` across cache API generations."""
    layers = getattr(past_kv, "layers", None)
    if layers is not None:
        for layer_idx, layer in enumerate(layers):
            initialized = getattr(layer, "is_initialized", True)
            if callable(initialized):
                initialized = initialized()
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if initialized and keys is not None and values is not None:
                if ReusableDynamicLayer is not None and isinstance(
                    layer, ReusableDynamicLayer
                ):
                    active = slice(0, layer.get_seq_length())
                    keys = keys[..., active, :]
                    values = values[..., active, :]
                yield layer_idx, keys, values
        return

    # transformers 4.x DynamicCache stores tensors in parallel lists.  Keep
    # support for it because GLiNER's dependency range spans both cache APIs.
    key_cache = getattr(past_kv, "key_cache", None)
    value_cache = getattr(past_kv, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        for layer_idx, (keys, values) in enumerate(zip(key_cache, value_cache)):
            if torch.is_tensor(keys) and torch.is_tensor(values):
                yield layer_idx, keys, values


def _dynamic_cache_from_layers(layer_values) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx, keys, values in layer_values:
        cache.update(keys, values, layer_idx)
    return cache


def _is_dynamic_cache_like(past_kv) -> bool:
    return hasattr(past_kv, "layers") or (
        hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache")
    )


def _past_kv_on_device(past_kv, device: torch.device) -> bool:
    if past_kv is None:
        return True
    if isinstance(past_kv, (tuple, list)):
        return all(_past_kv_on_device(item, device) for item in past_kv)
    cache_layers = list(_cache_layers(past_kv))
    return all(
        _device_matches(keys.device, device) and _device_matches(values.device, device)
        for _, keys, values in cache_layers
    )


def _move_past_kv(past_kv, device):
    if past_kv is None:
        return None
    if isinstance(past_kv, ReusableDynamicCache):
        return past_kv.move_to(device)
    if isinstance(past_kv, (tuple, list)):
        return type(past_kv)(_move_past_kv(item, device) for item in past_kv)

    cache_layers = list(_cache_layers(past_kv))
    if cache_layers or _is_dynamic_cache_like(past_kv):
        return _dynamic_cache_from_layers(
            (layer_idx, keys.to(device), values.to(device))
            for layer_idx, keys, values in cache_layers
        )
    if hasattr(past_kv, "to"):
        return past_kv.to(device)
    return past_kv


def _cache_from_layers(
    layer_values,
    *,
    config=None,
    max_cache_length: int | None = None,
) -> ReusableDynamicCache:
    """Build a reusable cache from complete per-layer key/value tensors."""
    cache = create_reusable_cache(config, max_cache_length=max_cache_length)
    for layer_idx, keys, values in layer_values:
        cache.update(keys, values, layer_idx)
    return cache


def copy_past_key_values(
    past_kv,
    *,
    config=None,
    max_cache_length: int | None = None,
):
    """Return an independent copy of a transformers-style KV cache.

    Streaming cache layers append in place.  Tests, transactional helpers, and
    dynamic batching therefore need an explicit copy operation rather than a
    shallow ``copy.copy`` of the cache container.
    """
    if past_kv is None:
        return None
    if torch.is_tensor(past_kv):
        return past_kv.detach().clone()
    if isinstance(past_kv, (tuple, list)):
        return type(past_kv)(
            copy_past_key_values(
                item,
                config=config,
                max_cache_length=max_cache_length,
            )
            for item in past_kv
        )

    cache_layers = list(_cache_layers(past_kv))
    if cache_layers or _is_dynamic_cache_like(past_kv):
        return _cache_from_layers(
            (
                (
                    layer_idx,
                    keys.detach().clone(),
                    values.detach().clone(),
                )
                for layer_idx, keys, values in cache_layers
            ),
            config=config,
            max_cache_length=max_cache_length,
        )
    raise TypeError(f"Unsupported past_key_values type: {type(past_kv).__name__}")


def stack_past_key_values(
    caches: Sequence[Any],
    *,
    config=None,
    max_cache_length: int | None = None,
) -> ReusableDynamicCache:
    """Stack equal-length, batch-size-one caches along their batch dimension."""
    if not caches:
        raise ValueError("At least one cache is required")

    cache_layers = [list(_cache_layers(cache)) for cache in caches]
    if any(not layers for layers in cache_layers):
        raise TypeError("Every cache must expose initialized key/value layers")

    layer_indices = [tuple(layer_idx for layer_idx, _, _ in layers) for layers in cache_layers]
    if any(indices != layer_indices[0] for indices in layer_indices[1:]):
        raise ValueError("Caches must contain the same initialized layers")

    stacked_layers = []
    for layer_offset, layer_idx in enumerate(layer_indices[0]):
        keys = [layers[layer_offset][1] for layers in cache_layers]
        values = [layers[layer_offset][2] for layers in cache_layers]
        reference_key_shape = keys[0].shape[1:]
        reference_value_shape = values[0].shape[1:]
        if any(key.size(0) != 1 or key.shape[1:] != reference_key_shape for key in keys):
            raise ValueError("Only compatible batch-size-one caches can be stacked")
        if any(value.size(0) != 1 or value.shape[1:] != reference_value_shape for value in values):
            raise ValueError("Only compatible batch-size-one caches can be stacked")
        stacked_layers.append(
            (
                layer_idx,
                torch.cat(keys, dim=0),
                torch.cat(values, dim=0),
            )
        )

    return _cache_from_layers(
        stacked_layers,
        config=config,
        max_cache_length=max_cache_length,
    )


def split_past_key_values(
    past_kv,
    lengths: Sequence[int],
    *,
    config=None,
    max_cache_length: int | None = None,
) -> list[ReusableDynamicCache]:
    """Split a batched cache into compact batch-size-one session caches."""
    normalized_lengths = [int(length) for length in lengths]
    if not normalized_lengths:
        return []
    if any(length < 0 for length in normalized_lengths):
        raise ValueError("Cache lengths must be non-negative")

    layers = list(_cache_layers(past_kv))
    if not layers:
        raise TypeError("past_key_values must expose initialized key/value layers")
    batch_size = layers[0][1].size(0)
    if len(normalized_lengths) != batch_size:
        raise ValueError("One cache length is required per batch row")

    results = []
    for row, length in enumerate(normalized_lengths):
        row_layers = []
        for layer_idx, keys, values in layers:
            if keys.size(0) != batch_size or values.size(0) != batch_size:
                raise ValueError("All cache layers must have the same batch size")
            if length > keys.size(-2) or length > values.size(-2):
                raise ValueError("Requested cache length exceeds the available cache")
            row_layers.append(
                (
                    layer_idx,
                    keys[row : row + 1, ..., :length, :].detach().clone(),
                    values[row : row + 1, ..., :length, :].detach().clone(),
                )
            )
        results.append(
            _cache_from_layers(
                row_layers,
                config=config,
                max_cache_length=max_cache_length,
            )
        )
    return results
