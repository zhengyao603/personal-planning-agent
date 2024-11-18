import os
import yaml
from typing import Dict
from typing import Any
from datetime import datetime

from langchain_core.tools import StructuredTool

def load_config() -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config.yaml')

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        return config

def current_time() -> str:
    """
    Obtains the current time in yyyy-mm-dd format
    """

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return time

def get_time_tool() -> StructuredTool:
    time_tool = StructuredTool.from_function(
        func=current_time,
        name="get_current_time",
        description="""Useful for when you need to get the current date and time.
        Note that the return is string in yyyy-mm-dd format, e.g. 2024-11-14.""",
    )
    return time_tool


if __name__ == '__main__':
    print(current_time())