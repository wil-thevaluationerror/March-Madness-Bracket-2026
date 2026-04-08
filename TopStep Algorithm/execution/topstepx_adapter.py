from __future__ import annotations

from execution.projectx_adapter import ProjectXAdapter


class TopstepXAdapter(ProjectXAdapter):
    """
    TopstepX-facing adapter.

    TopstepX's public API is backed by ProjectX, so this currently reuses the
    existing ProjectX mock/live surface while keeping the rest of the codebase
    expressed in TopstepX terms.
    """

