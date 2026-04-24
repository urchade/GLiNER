"""HTTP client for the GLiNER Ray Serve deployment.

This is a thin wrapper around the server's HTTP endpoint — it does not import
Ray and does not join the Ray cluster, so it works from any Python process
with only the standard library available.

To preserve server-side dynamic batching when multiple texts are submitted,
the client dispatches each text as its own HTTP request concurrently so Ray
Serve's ``@serve.batch`` can coalesce them into a single forward pass. A
sequential loop of calls would serialize on the wire and defeat batching.
"""

import json
from typing import Any, Dict, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_ROUTE_PREFIX = "/gliner"


class GLiNERClientError(RuntimeError):
    """Raised when the GLiNER server returns an error or is unreachable."""


class GLiNERClient:
    """HTTP client for a running GLiNER Ray Serve deployment.

    Example:
        >>> from gliner.serve import GLiNERClient
        >>> client = GLiNERClient()
        >>> client.predict(
        ...     "John works at Google in Mountain View",
        ...     labels=["person", "organization", "location"],
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
        self.url = base_url.rstrip("/") + "/" + route_prefix.strip("/")
        self.timeout = timeout
        self.max_concurrency = max_concurrency

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise GLiNERClientError(
                f"GLiNER server returned HTTP {e.code}: {body}"
            ) from e
        except URLError as e:
            raise GLiNERClientError(
                f"Could not reach GLiNER server at {self.url}: {e.reason}"
            ) from e

    @staticmethod
    def _build_payload(
        text: str,
        labels: List[str],
        relations: Optional[List[str]],
        threshold: Optional[float],
        relation_threshold: Optional[float],
        flat_ner: bool,
        multi_label: bool,
    ) -> Dict[str, Any]:
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
        return payload

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

        if len(payloads) == 1:
            results = [self._post(payloads[0])]
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
        """Async prediction. Concurrent calls coalesce into one server-side batch."""
        import asyncio

        single = isinstance(text, str)
        items = [text] if single else list(text)

        payloads = [
            self._build_payload(
                t, labels, relations, threshold, relation_threshold,
                flat_ner, multi_label,
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
