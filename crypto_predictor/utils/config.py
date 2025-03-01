import yaml
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    Args:
        config_path: Path to the YAML configuration file.
    Returns:
        Dictionary with configuration parameters.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info("Configuration loaded from %s", config_path)
        return config
    except Exception as e:
        logger.error("Configuration loading error: %s", e)
        raise
