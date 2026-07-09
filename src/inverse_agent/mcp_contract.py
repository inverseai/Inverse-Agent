"""Lightweight MCP contract description.

The concrete MCP server process is intentionally deferred until an adapter needs
network/process transport. The product-facing contract is still typed here so
adapters expose stable tool metadata and structured calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class McpToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    safety: str


@dataclass
class McpCallResult:
    ok: bool
    content: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

