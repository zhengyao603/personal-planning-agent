import os
import yaml
from typing import Dict
from typing import Any

def load_config() -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config.yaml')

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        return config