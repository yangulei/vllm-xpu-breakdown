# SPDX-License-Identifier: Apache-2.0
"""Vocabulary shared by every stage of the pipeline.

The stages disagreed before this package existed: five separate dtype tables,
~28 op-name tables and three symbolic-dimension parsers each encoded the same
fact in a slightly different way, and some of them had drifted apart (the
benchmark called ``float16`` ``f16`` while the shape matrix called it ``fp16``;
the browser and the server resolved ``S+C`` differently).

Everything here is a *fact about a name*, not a computation, and is kept
torch-free so the offline paths -- reconstruction from an uploaded trace, the
shape matrix on a GPU-less box -- import it cheaply.
"""
