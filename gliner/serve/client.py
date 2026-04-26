"""Client for accessing GLiNER Ray Serve deployment from Python."""

from typing import Any, Dict, List, Union, Optional


class GLiNERClient:
    """Client for accessing a running GLiNER Ray Serve deployment.

    This client can be used from any Python process (same machine or remote)
    to make predictions against a running GLiNER server.

    Example:
        >>> from gliner.serve import GLiNERClient
        >>> client = GLiNERClient()
        >>> results = client.predict(
        ...     "John works at Google in Mountain View", labels=["person", "organization", "location"]
        ... )
        >>> print(results)
        {'entities': [{'start': 0, 'end': 4, 'text': 'John', 'label': 'person', 'score': 0.95}, ...]}
    """

    def __init__(
        self,
        deployment_name: str = "gliner",
        ray_address: Optional[str] = None,
    ):
        """Initialize client connection to GLiNER deployment.

        Args:
            deployment_name: Name of the Ray Serve deployment.
            ray_address: Ray cluster address. If None, connects to local cluster.
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
        """Make predictions using the GLiNER server.

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
            ref = self._handle.predict.remote(
                text, labels, relations, threshold, relation_threshold, flat_ner, multi_label
            )
            return ref.result()

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


def get_client(
    deployment_name: str = "gliner",
    ray_address: Optional[str] = None,
) -> GLiNERClient:
    """Get a client for the GLiNER Ray Serve deployment.

    Args:
        deployment_name: Name of the Ray Serve deployment.
        ray_address: Ray cluster address. If None, connects to local cluster.

    Returns:
        GLiNERClient instance for making predictions.
    """
    return GLiNERClient(deployment_name, ray_address)
