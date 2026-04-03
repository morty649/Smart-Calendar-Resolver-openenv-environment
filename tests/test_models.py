import pytest
from pydantic import ValidationError

from calender_en.models import CalenderEnAction, CalenderEnObservation


def test_action_model_validates_expected_fields() -> None:
    action = CalenderEnAction(
        stage="propose_slot",
        proposed_time_slot="2026-04-08 10:00-10:30 UTC",
        confirm_schedule=False,
        final_note="Pick the earliest shared slot.",
    )

    assert action.stage == "propose_slot"
    assert action.proposed_time_slot == "2026-04-08 10:00-10:30 UTC"


def test_action_model_rejects_invalid_stage() -> None:
    with pytest.raises(ValidationError):
        CalenderEnAction(stage="invalid_stage", final_note="bad")


def test_observation_model_validates_expected_fields() -> None:
    observation = CalenderEnObservation(
        request="Schedule a sync.",
        availability={"Alex": ["2026-04-08 10:00-10:30 UTC"]},
        constraints={"duration": "30 minutes", "priority": "high", "deadline": "2026-04-09"},
        step_count=1,
        reward=1.0,
        done=False,
        feedback="Looks good.",
        next_expected_stage="evaluate_availability",
    )

    assert observation.request == "Schedule a sync."
    assert observation.next_expected_stage == "evaluate_availability"
