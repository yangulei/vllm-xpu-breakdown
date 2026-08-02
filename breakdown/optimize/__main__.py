# SPDX-License-Identifier: Apache-2.0
"""``python -m breakdown.optimize`` entry point."""
from __future__ import annotations

import sys

from breakdown.optimize.cli import main

if __name__ == "__main__":
    sys.exit(main())
