"""HTTP client for the GLiNER Ray Serve deployment.

from typing import Any, Dict, List, Union, Optional

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_ROUTE_PREFIX = "/gliner"


class GLiNERClientError(RuntimeError):
    """Raised when the GLiNER server returns an error or is unreachable."""


class GLiNERClient:
    """HTTP client for a running GLiNER Ray Serve deployment.

    Example:
        >>> from gliner.serve import GLiNERClient
        >>> client = GLiNERClient()
        >>> results = client.predict(
        ...     "John works at Google in Mountain View", labels=["person", "organization", "location"]
        ... )
        {'entities': [{'start': 0, 'end': 4, 'text': 'John', 'label': 'person', ...}, ...]}
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        route_prefix: str = DEFAULT_ROUTE_PREFIX,
        timeout: float = 30.0,
        max_concurrency: int = 32,
    ):
        """Initialize the HTTP client.

        Args:
            base_url: Scheme + host + port of the Ray Serve HTTP proxy.
            route_prefix: Route prefix the deployment is mounted under (must
                match ``GLiNERServeConfig.route_prefix``).
            timeout: Per-request timeout in seconds.
            max_concurrency: Maximum in-flight HTTP requests when predicting
                on a list of texts. Bounds the client-side thread pool.
        """
        import ray  # noqa: PLC0415
        from ray import serve  # noqa: PLC0415

        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)

        self._handle = serve.get_deployment_handle(deployment_name, "gliner")

    def predict(
        self,
        text: Union[str, List[str]],
        labels: List[str],
        relations: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        relation_threshold: Optional[float] = None,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Blocking prediction. ``str`` in → ``dict`` out; ``list`` in → ``list`` out."""
        single = isinstance(text, str)
        items = [text] if single else list(text)

        payloads = [
            self._build_payload(
                t, labels, relations, threshold, relation_threshold,
                flat_ner, multi_label,
            )
            for t in items
        ]

        Args:
            text: Input text or list of texts.
            labels: Entity type labels to extract.
            relations: Relation type labels (for relex models).
            threshold: Confidence threshold for entities.
            relation_threshold: Confidence threshold for relations.
            flat_ner: Whether to use flat NER.
            multi_label: Whether to allow multiple labels per span.

        Returns:
            Single result dict or list of result dicts containing:
                - "entities": List of entity dicts
                - "relations": List of relation dicts (if model supports)
        """
        if isinstance(text, list):
            refs = [
                self._handle.predict.remote(t, labels, relations, threshold, relation_threshold, flat_ner, multi_label)
                for t in text
            ]
            return [ref.result() for ref in refs]
        else:
            workers = min(self.max_concurrency, len(payloads))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(self._post, payloads))

        return results[0] if single else results

    async def predict_async(
        self,
        text: Union[str, List[str]],
        labels: List[str],
        relations: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        relation_threshold: Optional[float] = None,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Async version of predict."""
        if isinstance(text, list):
            import asyncio  # noqa: PLC0415

            refs = [
                self._handle.predict.remote(t, labels, relations, threshold, relation_threshold, flat_ner, multi_label)
                for t in text
            ]
            results = await asyncio.gather(*refs)
            return list(results)
        else:
            return await self._handle.predict.remote(
                text, labels, relations, threshold, relation_threshold, flat_ner, multi_label
            )
            for t in items
        ]

        results = await asyncio.gather(
            *(asyncio.to_thread(self._post, p) for p in payloads)
        )

        return results[0] if single else list(results)


def get_client(
    base_url: str = DEFAULT_BASE_URL,
    route_prefix: str = DEFAULT_ROUTE_PREFIX,
    timeout: float = 30.0,
    max_concurrency: int = 32,
) -> GLiNERClient:
    """Convenience constructor for :class:`GLiNERClient`."""
    return GLiNERClient(
        base_url=base_url,
        route_prefix=route_prefix,
        timeout=timeout,
        max_concurrency=max_concurrency,
    )
