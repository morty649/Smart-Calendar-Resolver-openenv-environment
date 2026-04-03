from calender_en.models import CalenderEnAction
from calender_en.server.calender_en_environment import CalenderEnEnvironment


def _advance_to_proposal(env: CalenderEnEnvironment) -> None:
    env.step(
        CalenderEnAction(
            stage="understand_request",
            final_note="Identify participants, objective, and deadline.",
        )
    )
    env.step(
        CalenderEnAction(
            stage="evaluate_availability",
            final_note="Intersect shared availability and choose the earliest feasible option.",
        )
    )


def test_reset_returns_valid_initial_observation() -> None:
    env = CalenderEnEnvironment()

    observation = env.reset()

    assert observation.request
    assert observation.availability
    assert observation.constraints
    assert observation.step_count == 0
    assert observation.reward == 0.0
    assert observation.done is False
    assert observation.next_expected_stage == "understand_request"
    assert env.state.step_count == 0
    assert env.history == []
    assert env.solved is False


def test_step_follows_expected_multi_stage_flow() -> None:
    env = CalenderEnEnvironment()
    env.reset()

    understand = env.step(
        CalenderEnAction(
            stage="understand_request",
            final_note="Identify participants, objective, and deadline.",
        )
    )
    availability = env.step(
        CalenderEnAction(
            stage="evaluate_availability",
            final_note="Intersect shared availability and prioritize the earliest option.",
        )
    )
    proposal = env.step(
        CalenderEnAction(
            stage="propose_slot",
            proposed_time_slot="2026-04-08 10:00-10:30 UTC",
            final_note="Pick the earliest shared slot before the deadline.",
        )
    )
    confirmation = env.step(
        CalenderEnAction(
            stage="confirm_schedule",
            proposed_time_slot="2026-04-08 10:00-10:30 UTC",
            confirm_schedule=True,
            final_note="Confirmed and invite is ready.",
        )
    )

    assert understand.next_expected_stage == "evaluate_availability"
    assert availability.next_expected_stage == "propose_slot"
    assert proposal.next_expected_stage == "confirm_schedule"
    assert confirmation.done is True
    assert confirmation.next_expected_stage is None
    assert env.solved is True


def test_deterministic_scenario_cycling() -> None:
    env = CalenderEnEnvironment()

    requests = [env.reset().request for _ in range(4)]

    assert requests[0] != requests[1]
    assert requests[1] != requests[2]
    assert requests[0] == requests[3]


def test_correct_slot_scores_higher_than_wrong_slot() -> None:
    correct_env = CalenderEnEnvironment()
    correct_env.reset()
    _advance_to_proposal(correct_env)
    correct = correct_env.step(
        CalenderEnAction(
            stage="propose_slot",
            proposed_time_slot="2026-04-08 10:00-10:30 UTC",
            final_note="Pick the earliest shared slot before the deadline.",
        )
    )

    wrong_env = CalenderEnEnvironment()
    wrong_env.reset()
    _advance_to_proposal(wrong_env)
    wrong = wrong_env.step(
        CalenderEnAction(
            stage="propose_slot",
            proposed_time_slot="2026-04-08 14:00-14:30 UTC",
            final_note="Pick a slot even if it is not shared.",
        )
    )

    assert correct.reward > wrong.reward


def test_state_updates_episode_id_step_count_history_and_solved() -> None:
    env = CalenderEnEnvironment()

    first_reset = env.reset()
    first_episode_id = env.state.episode_id
    assert first_reset.step_count == 0
    assert env.history == []
    assert env.solved is False

    env.step(
        CalenderEnAction(
            stage="understand_request",
            final_note="Identify participants, objective, and deadline.",
        )
    )
    assert env.state.step_count == 1
    assert len(env.history) == 1
    assert env.history[0]["stage"] == "understand_request"
    assert env.solved is False

    env.step(
        CalenderEnAction(
            stage="evaluate_availability",
            final_note="Intersect shared availability and prioritize the earliest option.",
        )
    )
    env.step(
        CalenderEnAction(
            stage="propose_slot",
            proposed_time_slot="2026-04-08 10:00-10:30 UTC",
            final_note="Pick the earliest shared slot before the deadline.",
        )
    )
    env.step(
        CalenderEnAction(
            stage="confirm_schedule",
            proposed_time_slot="2026-04-08 10:00-10:30 UTC",
            confirm_schedule=True,
            final_note="Confirmed and invite is ready.",
        )
    )

    assert env.state.step_count == 4
    assert len(env.history) == 4
    assert env.solved is True

    env.reset()
    assert env.state.episode_id != first_episode_id
    assert env.state.step_count == 0
    assert env.history == []
    assert env.solved is False
