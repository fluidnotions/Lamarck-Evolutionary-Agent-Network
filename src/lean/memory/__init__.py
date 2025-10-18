"""
Memory management with timestamped decay.
"""

# Import MemoryManager from parent module (memory.py file)
import sys
from pathlib import Path

# Add parent directory to path to import memory.py
parent_path = Path(__file__).parent.parent
if str(parent_path) not in sys.path:
    sys.path.insert(0, str(parent_path))

# Import from sibling memory.py module
try:
    # When memory.py exists as sibling
    import importlib.util
    memory_file = Path(__file__).parent.parent / "memory.py"
    if memory_file.exists():
        spec = importlib.util.spec_from_file_location("lean_memory_module", memory_file)
        memory_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(memory_module)
        MemoryManager = memory_module.MemoryManager
        generate_collection_name = memory_module.generate_collection_name
    else:
        # Fallback - MemoryManager might be defined elsewhere
        MemoryManager = None
        generate_collection_name = None
except Exception:
    MemoryManager = None
    generate_collection_name = None

__all__ = ["DecayCalculator", "MemoryPruner", "MemoryManager", "generate_collection_name"]
