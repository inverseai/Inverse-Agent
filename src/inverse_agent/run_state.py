"""Centralized durable run-state transitions.

Stores use this table instead of inventing one-off compare-and-swap edges.  It
is intentionally explicit: adding a status is a reviewable state-machine
change, not an accidental consequence of a broad SQL update.
"""

from __future__ import annotations

from inverse_agent.models import RunStatus

TERMINAL_STATUSES = frozenset(
    {
        RunStatus.SUCCEEDED,
        RunStatus.INCOMPLETE,
        RunStatus.CANCELLED,
        RunStatus.FAILED,
        RunStatus.REFUSED,
    }
)

LEGAL_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PLANNED: frozenset({RunStatus.QUEUED, RunStatus.CANCELLED}),
    RunStatus.QUEUED: frozenset(
        {
            RunStatus.STARTING,
            RunStatus.APPROVING,
            RunStatus.CANCELLED,
            RunStatus.INCOMPLETE,
            RunStatus.FAILED,
        }
    ),
    RunStatus.STARTING: frozenset(
        {
            RunStatus.QUEUED,
            RunStatus.RUNNING,
            RunStatus.WAITING_FOR_APPROVAL,
            RunStatus.CANCEL_REQUESTED,
            RunStatus.SUCCEEDED,
            RunStatus.INCOMPLETE,
            RunStatus.CANCELLED,
            RunStatus.FAILED,
            RunStatus.REFUSED,
        }
    ),
    RunStatus.APPROVING: frozenset(
        {
            RunStatus.QUEUED,
            RunStatus.RUNNING,
            RunStatus.WAITING_FOR_APPROVAL,
            RunStatus.CANCEL_REQUESTED,
            RunStatus.SUCCEEDED,
            RunStatus.INCOMPLETE,
            RunStatus.CANCELLED,
            RunStatus.FAILED,
            RunStatus.REFUSED,
        }
    ),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.QUEUED,
            RunStatus.WAITING_FOR_APPROVAL,
            RunStatus.CANCEL_REQUESTED,
            RunStatus.SUCCEEDED,
            RunStatus.INCOMPLETE,
            RunStatus.CANCELLED,
            RunStatus.FAILED,
            RunStatus.REFUSED,
        }
    ),
    RunStatus.WAITING_FOR_APPROVAL: frozenset(
        {
            RunStatus.QUEUED,
            RunStatus.CANCELLED,
            RunStatus.INCOMPLETE,
            RunStatus.REFUSED,
            RunStatus.FAILED,
        }
    ),
    RunStatus.CANCEL_REQUESTED: frozenset({RunStatus.CANCELLED}),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.INCOMPLETE: frozenset(),
    RunStatus.CANCELLED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.REFUSED: frozenset(),
}


def require_transition(current: RunStatus, target: RunStatus) -> None:
    """Raise when a requested durable transition is not part of the contract."""

    if target not in LEGAL_TRANSITIONS[current]:
        raise ValueError(f"illegal run status transition: {current.value} -> {target.value}")


def is_terminal(status: str | RunStatus) -> bool:
    try:
        parsed = status if isinstance(status, RunStatus) else RunStatus(status)
    except ValueError:
        return False
    return parsed in TERMINAL_STATUSES
