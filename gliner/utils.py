import json
import argparse
from typing import Any, Dict, Union
from pathlib import Path

import yaml


def load_config_as_namespace(config_file: Union[str, Path]) -> argparse.Namespace:
    """
    Load YAML/JSON config file as nested Namespace.

    Args:
        config_file: Path to config file

    Returns:
        Nested argparse.Namespace

    Example:
        >>> config = load_config_as_namespace("config.yaml")
        >>> print(config.model.model_name)
        >>> print(config.training.lr_encoder)
    """
    config_file = Path(config_file)

    with open(config_file) as f:
        if config_file.suffix in {".yml", ".yaml"}:
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)

    return dict_to_namespace(config_dict)


def dict_to_namespace(d: Dict[str, Any]) -> argparse.Namespace:
    """Recursively convert dict to Namespace."""
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        elif isinstance(value, list):
            setattr(namespace, key, [dict_to_namespace(item) if isinstance(item, dict) else item for item in value])
        else:
            setattr(namespace, key, value)
    return namespace


def namespace_to_dict(namespace: argparse.Namespace) -> Dict[str, Any]:
    """Convert Namespace back to dict."""
    result = {}
    for key, value in vars(namespace).items():
        if isinstance(value, argparse.Namespace):
            result[key] = namespace_to_dict(value)
        elif isinstance(value, list):
            result[key] = [namespace_to_dict(item) if isinstance(item, argparse.Namespace) else item for item in value]
        else:
            result[key] = value
    return result


def is_module_available(module_name):
    """
    Checks whether the specified Python module is available.

    Args:
        module_name (str): The name of the module to check.

    Returns:
        bool: True if the module is available, False otherwise.
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


class MissedPackageException(Exception):
    """Raised when the requested decoder model is not supported."""

    pass
