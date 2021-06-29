from importlib import import_module
from pathlib import Path

__all__ = [
    import_module(f".{f.stem}", __package__)
    for f in Path(__file__).parent.glob("*.py")
    if "__" not in f.stem
]
del import_module, Path
