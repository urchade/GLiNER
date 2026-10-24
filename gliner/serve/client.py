"""HTTP client for the GLiNER Ray Serve deployment."""

from typing import Any, Dict, List

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_ROUTE_PREFIX = "/gliner"

LabelSet = List[str] | Dict[str, str]
Labels = LabelSet | List[LabelSet]


def _labels_for_texts(labels: Labels, num_texts: int) -> List[LabelSet]:
    """Return one label set per text while preserving shared label sets."""
    if isinstance(labels, list) and labels and isinstance(labels[0], (list, dict)):
        if len(labels) != num_texts:
            raise ValueError(f"Per-text labels must have length {num_texts}, got {len(labels)}")
        return labels
    return [labels] * num_texts


class GLiNERClientError(RuntimeError):
    """Raised when the GLiNER server returns an error or is unreachable."""


class GLiNERAPIError(GLiNERClientError):
    """The analyze API rejected a request with an error object (``code``, ``message``, ``path``)."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int | None = None,
        path: str | None = None,
        details: Dict[str, Any] | None = None,
    ):
        super().__init__(f"{code}: {message}" + (f" (at {path})" if path else ""))
        self.code = code
        self.message = message
        self.status = status
        self.path = path
        self.details = details

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], status: int | None = None) -> "GLiNERAPIError":
        error = payload.get("error") or {}
        return cls(
            error.get("code", "INTERNAL"),
            error.get("message", ""),
            status=status,
            path=error.get("path"),
            details=error.get("details"),
        )


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
        model: str | None = None,
    ):
        """Initialize the HTTP client.

        Args:
            base_url: Scheme + host + port of the Ray Serve HTTP proxy.
            route_prefix: Route prefix the deployment is mounted under (must
                match ``GLiNERServeConfig.route_prefix``).
            timeout: Per-request timeout in seconds.
            max_concurrency: Maximum in-flight HTTP requests when predicting
                on a list of texts. Bounds the client-side thread pool.
            model: Model id for analyze requests. Defaults to the first id
                ``GET /v1/models`` reports.
        """
        self.url = base_url.rstrip("/") + route_prefix
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self._model = model

    def _build_payload(
        self,
        text: str,
        labels: LabelSet,
        relations: List[str] | None,
        threshold: float | None,
        relation_threshold: float | None,
        flat_ner: bool,
        multi_label: bool,
        adapter_id: str | None,
    ) -> Dict[str, Any]:
        """Build the JSON payload for a single prediction request."""
        payload: Dict[str, Any] = {
            "text": text,
            "labels": labels,
            "flat_ner": flat_ner,
            "multi_label": multi_label,
        }
        if relations is not None:
            payload["relations"] = relations
        if threshold is not None:
            payload["threshold"] = threshold
        if relation_threshold is not None:
            payload["relation_threshold"] = relation_threshold
        if adapter_id is not None:
            payload["adapter_id"] = adapter_id
        return payload

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a single POST request to the server."""
        import json  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            raise GLiNERClientError(f"Request to {self.url} failed: {exc}") from exc

    def predict(
        self,
        text: str | List[str],
        labels: Labels,
        relations: List[str] | None = None,
        threshold: float | None = None,
        relation_threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        adapter_id: str | None = None,
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Blocking prediction. ``str`` in -> ``dict`` out; ``list`` in -> ``list`` out."""
        single = isinstance(text, str)
        items = [text] if single else list(text)
        labels_list = _labels_for_texts(labels, len(items))

        payloads = [
            self._build_payload(
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

        if len(payloads) == 1:
            results = [self._post(payloads[0])]
        else:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

            workers = min(self.max_concurrency, len(payloads))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(self._post, payloads))

        return results[0] if single else results

    async def predict_async(
        self,
        text: str | List[str],
        labels: Labels,
        relations: List[str] | None = None,
        threshold: float | None = None,
        relation_threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        adapter_id: str | None = None,
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Async version of predict."""
        import asyncio  # noqa: PLC0415

        single = isinstance(text, str)
        items = [text] if single else list(text)
        labels_list = _labels_for_texts(labels, len(items))

        payloads = [
            self._build_payload(
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

        results = await asyncio.gather(*(asyncio.to_thread(self._post, p) for p in payloads))

        return results[0] if single else list(results)

    def adapter_cache_status(self, adapter_id: str | None = None) -> Dict[str, Any]:
        """Return PolyLoRA adapter cache status from the server."""
        import json  # noqa: PLC0415
        import urllib.parse  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        url = f"{self.url}/adapter-cache"
        if adapter_id is not None:
            url += "?" + urllib.parse.urlencode({"adapter_id": adapter_id})
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            raise GLiNERClientError(f"Request to {url} failed: {exc}") from exc

    def is_adapter_cached(self, adapter_id: str) -> bool:
        return bool(self.adapter_cache_status(adapter_id).get("cached"))

    # -- analyze API (``/v1``) ---------------------------------------------------------

    def _request(self, method: str, suffix: str, payload: Any = None) -> Any:
        """Call ``{url}/v1{suffix}``; API error objects raise ``GLiNERAPIError``."""
        import json  # noqa: PLC0415
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        url = f"{self.url}/v1{suffix}"
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
                raise GLiNERAPIError.from_payload(parsed, exc.code) from exc
            raise GLiNERClientError(f"Request to {url} failed with HTTP {exc.code}: {body}") from exc
        except Exception as exc:
            raise GLiNERClientError(f"Request to {url} failed: {exc}") from exc

    @property
    def model(self) -> str:
        """Model id sent with analyze requests: the one given, else the server's first model."""
        if self._model is None:
            self._model = self.models()["data"][0]["id"]
        return self._model

    @model.setter
    def model(self, value: str | None) -> None:
        self._model = value

    def _analyze_body(self, input, tasks, options, model) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model or self.model, "input": input, "tasks": tasks}
        if options is not None:
            body["options"] = options
        return body

    def _batch_body(self, items, options, model) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model or self.model, "items": items}
        if options is not None:
            body["options"] = options
        return body

    def analyze(
        self,
        input: Any,
        tasks: Dict[str, Dict[str, Any]],
        *,
        options: Dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Dict[str, Any]:
        """``POST /v1/analyze``: one input, many named ``span`` / ``relation`` tasks.

        Example:
            >>> client.analyze(
            ...     "Alice founded Acme in London.",
            ...     tasks={
            ...         "entities": {"type": "span", "labels": ["person", "organization"], "threshold": 0.4},
            ...         "places": {"type": "span", "labels": "location", "flat_ner": False},
            ...     },
            ... )["results"]["entities"]["value"]
        """
        return self._request("POST", "/analyze", self._analyze_body(input, tasks, options, model))

    def analyze_batch(
        self,
        items: List[Dict[str, Any]],
        *,
        options: Dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Dict[str, Any]:
        """``POST /v1/analyze/batch``: items are ``{"id"?, "input", "tasks", "options"?}``."""
        return self._request("POST", "/analyze/batch", self._batch_body(items, options, model))

    async def analyze_async(
        self,
        input: Any,
        tasks: Dict[str, Dict[str, Any]],
        *,
        options: Dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Dict[str, Any]:
        import asyncio  # noqa: PLC0415

        return await asyncio.to_thread(self.analyze, input, tasks, options=options, model=model)

    async def analyze_batch_async(
        self,
        items: List[Dict[str, Any]],
        *,
        options: Dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Dict[str, Any]:
        import asyncio  # noqa: PLC0415

        return await asyncio.to_thread(self.analyze_batch, items, options=options, model=model)

    def extract_spans(
        self,
        input: Any,
        labels: Any,
        *,
        threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        adapter_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """One ``span`` task; returns its list of spans."""
        task: Dict[str, Any] = {"type": "span", "labels": labels, "flat_ner": flat_ner, "multi_label": multi_label}
        if threshold is not None:
            task["threshold"] = threshold
        options = {"adapter_id": adapter_id} if adapter_id is not None else None
        return self.analyze(input, {"spans": task}, options=options)["results"]["spans"]["value"]

    def extract_relations(
        self,
        input: Any,
        labels: Any,
        entity_labels: Any,
        *,
        threshold: float | None = None,
        entity_threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        adapter_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """One ``relation`` task; returns its list of relations."""
        task: Dict[str, Any] = {
            "type": "relation",
            "labels": labels,
            "entity_labels": entity_labels,
            "flat_ner": flat_ner,
            "multi_label": multi_label,
        }
        if threshold is not None:
            task["threshold"] = threshold
        if entity_threshold is not None:
            task["entity_threshold"] = entity_threshold
        options = {"adapter_id": adapter_id} if adapter_id is not None else None
        return self.analyze(input, {"relations": task}, options=options)["results"]["relations"]["value"]

    def models(self) -> Dict[str, Any]:
        """``GET /v1/models``."""
        return self._request("GET", "/models")

    def capabilities(self, model: str | None = None) -> Dict[str, Any]:
        """``GET /v1/models/{model}/capabilities``."""
        return self._request("GET", f"/models/{model or self.model}/capabilities")

    def health(self) -> Dict[str, Any]:
        """``GET /v1/health``."""
        return self._request("GET", "/health")


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
