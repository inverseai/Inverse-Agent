from pathlib import Path

from inverse_agent.models import AutonomyLevel, Domain, WorkspaceProfile


def test_workspace_autonomy_is_per_domain() -> None:
    profile = WorkspaceProfile(
        root=Path("."),
        domains={Domain.DJANGO, Domain.IOS},
        autonomy={Domain.DJANGO: AutonomyLevel.BOUNDED_AUTO},
    )

    assert profile.autonomy_for(Domain.DJANGO) == AutonomyLevel.BOUNDED_AUTO
    assert profile.autonomy_for(Domain.IOS) == AutonomyLevel.ASSISTED

