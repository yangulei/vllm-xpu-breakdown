# SPDX-License-Identifier: Apache-2.0
"""``python -m breakdown.bench`` entry point."""
from __future__ import annotations

import sys

from breakdown.bench.cli import main

if __name__ == "__main__":
    sys.exit(main())
