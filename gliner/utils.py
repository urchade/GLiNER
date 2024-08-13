import argparse
import yaml

def load_config_as_namespace(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)

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
