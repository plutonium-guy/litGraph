"""Public entry point for `python -m litgraph.doctor`.

Forwards to `_doctor.main`. Kept as a thin shim so `_doctor.py` can be
imported without triggering CLI dispatch.
"""
from ._doctor import main


if __name__ == "__main__":
    import sys
    sys.exit(main())
