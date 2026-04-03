# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Calender En Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CalenderEnAction, CalenderEnObservation


class CalenderEnEnv(
    EnvClient[CalenderEnAction, CalenderEnObservation, State]
):
    """
    Client for the Calender En Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CalenderEnEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.request)
        ...
        ...     result = client.step(
        ...         CalenderEnAction(stage="understand_request", final_note="Identify participants and deadline.")
        ...     )
        ...     print(result.observation.feedback)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CalenderEnEnv.from_docker_image("calender_en-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(
        ...         CalenderEnAction(stage="understand_request", final_note="Identify participants and deadline.")
        ...     )
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CalenderEnAction) -> Dict:
        """
        Convert CalenderEnAction to JSON payload for step message.

        Args:
            action: CalenderEnAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "stage": action.stage,
            "proposed_time_slot": action.proposed_time_slot,
            "confirm_schedule": action.confirm_schedule,
            "final_note": action.final_note,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CalenderEnObservation]:
        """
        Parse server response into StepResult[CalenderEnObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CalenderEnObservation
        """
        obs_data = payload.get("observation", {})
        observation = CalenderEnObservation(
            request=obs_data.get("request", ""),
            availability=obs_data.get("availability", {}),
            constraints=obs_data.get("constraints", {}),
            step_count=obs_data.get("step_count", 0),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            done=payload.get("done", False),
            feedback=obs_data.get("feedback", ""),
            next_expected_stage=obs_data.get("next_expected_stage"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
