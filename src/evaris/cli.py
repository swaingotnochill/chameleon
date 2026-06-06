"""chameleon / evaris CLI — thin wrapper around examples/workflow.py."""

import asyncio
import sys
from pathlib import Path

# Ensure examples/ is importable
_EXAMPLES = str(Path(__file__).resolve().parent.parent.parent / "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

from workflow import main as _main  # noqa: E402


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
