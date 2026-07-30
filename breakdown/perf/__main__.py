# SPDX-License-Identifier: Apache-2.0
"""``python -m breakdown.perf`` entry point."""
from __future__ import annotations

from breakdown.perf.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
