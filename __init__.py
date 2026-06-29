"""
ComfyUI-MotionStreamer

ComfyUI nodes for MotionStreamer text-to-motion generation.
"""

import importlib.util
import os
from pathlib import Path

if not os.environ.get("PYTEST_CURRENT_TEST"):
    print("[ComfyUI-MotionStreamer] Initializing custom node...")

try:
    nodes_path = Path(__file__).with_name("nodes.py")
    spec = importlib.util.spec_from_file_location("comfyui_motionstreamer_nodes", nodes_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    NODE_CLASS_MAPPINGS = module.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = module.NODE_DISPLAY_NAME_MAPPINGS
    print("[ComfyUI-MotionStreamer] [OK] Node classes imported successfully")
except Exception as e:
    import traceback

    print(f"[ComfyUI-MotionStreamer] [ERROR] Failed to import node classes: {e}")
    print(f"[ComfyUI-MotionStreamer] Traceback:\n{traceback.format_exc()}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__version__ = "0.1.0"
