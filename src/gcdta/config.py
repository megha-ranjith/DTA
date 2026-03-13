"""Configuration loading utilities for innovations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None


class ConfigLoader:
    """Load YAML configuration files with inheritance support."""

    def __init__(self, config_dir: Path = Path(__file__).parent.parent.parent / "configs"):
        self.config_dir = config_dir

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load config file with inheritance support.

        Args:
            config_name: Config filename (e.g., 'path1_pocket_uncertainty.yaml' or 'configs/base.yaml')

        Returns:
            Merged configuration dictionary
        """
        if yaml is None:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")

        # Handle both 'base.yaml' and 'configs/base.yaml' formats
        if config_name.startswith("configs/"):
            config_name = config_name[8:]  # Strip 'configs/' prefix
        
        config_path = self.config_dir / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Handle inheritance
        if "extends" in config:
            base_name = config.pop("extends")
            base_config = self.load(base_name)

            # Merge: child overrides parent
            config = self._merge_dicts(base_config, config)

        return config

    @staticmethod
    def _merge_dicts(base: Dict, override: Dict) -> Dict:
        """Recursively merge override into base."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
