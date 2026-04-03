---
title: SmartCalendarResolver
emoji: "📅"
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - scheduling
---

# SmartCalendarResolver

SmartCalendarResolver is a deterministic OpenEnv scheduling environment. Each episode walks through a fixed four-stage workflow:

1. understand request
2. evaluate availability
3. propose slot
4. confirm schedule

The environment uses a small in-code dataset with easy, medium, and hard scenarios. There is no randomness.

## Action Model

`CalenderEnAction` fields:

- `stage`: one of `understand_request`, `evaluate_availability`, `propose_slot`, `confirm_schedule`
- `proposed_time_slot`: optional slot string used during proposal or confirmation
- `confirm_schedule`: boolean used only during confirmation
- `final_note`: short reasoning or confirmation note

## Observation Model

`CalenderEnObservation` fields:

- `request`
- `availability`
- `constraints`
- `step_count`
- `reward`
- `done`
- `feedback`
- `next_expected_stage`

## Reward Behavior

- Correct stage ordering is rewarded.
- Correct slot selection is rewarded.
- Respecting deadline and earliest-slot constraints is rewarded.
- Proper final confirmation is rewarded.
- Invalid or repeated actions are penalized.

## Local Development

Install dependencies:

```bash
uv sync
```

Validate the environment:

```bash
uv run openenv validate .
```

Run the deterministic baseline:

```bash
uv run python inference.py
```

Start the FastAPI server:

```bash
uv run python server/app.py
```

Health check:

```bash
curl http://localhost:8000/health
```

## Docker

Build:

```bash
docker build -t smart-calendar-env .
```

Run:

```bash
docker run -p 8000:8000 smart-calendar-env
```

Then verify:

```bash
curl http://localhost:8000/health
```
