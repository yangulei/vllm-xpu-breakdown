# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration.

Puts the repository root on ``sys.path`` (so ``breakdown``, ``app`` and
``tests.data`` import the same way from any working directory) and defines the
``--update-golden`` flag used by the snapshot tests.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden", action="store_true", default=False,
        help="rewrite the golden snapshots in tests/data/golden/ from the "
             "current output instead of asserting against them",
    )
