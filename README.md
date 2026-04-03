# Smart Calendar Resolver — OpenEnv Environment

A deterministic, multi-step OpenEnv environment for evaluating agent reasoning in real-world scheduling workflows.

This environment models a constrained meeting scheduling problem where an agent must interpret user intent, reason over structured availability, and produce a valid, verified outcome through a staged interaction loop.

---

## Problem Definition

Given:
- a natural language meeting request
- multiple participants with availability windows
- constraints (duration, deadline, priority, timezone)

The agent must:
1. Interpret the request
2. Aggregate and reason over availability
3. Select a valid time slot
4. Confirm and finalize the schedule

This reflects real-world calendar coordination tasks commonly handled by assistants and productivity tools.

---

## Environment Design

### Core Loop

The environment follows the standard OpenEnv interface:

- `reset()` → returns initial observation
- `step(action)` → returns (observation, reward, done, info)
- `state` → internal environment state

### Stage-Based Interaction

The task is decomposed into explicit stages:

1. `understand_request`
2. `evaluate_availability`
3. `propose_slot`
4. `confirm_schedule`

Agents are expected to follow this progression. Out-of-order or invalid transitions are penalized.

---

## Dataset

A small, fully deterministic, in-memory dataset is used.

Each scenario includes:
- request text
- participants
- availability windows
- constraints (deadline, duration, priority)
- ground-truth valid slot

Difficulty levels:
- **Easy**: single valid slot, minimal reasoning
- **Medium**: conflicting availability with constraint filtering
- **Hard**: multiple candidates requiring prioritization and constraint trade-offs

Design choice:
- Small dataset ensures reproducibility
- No randomness ensures stable evaluation and debugging

---

## State Representation

The environment maintains:

- `episode_id`
- `step_count`
- `current_scenario`
- `selected_slot`
- `action_history`
- `solved` flag

This enables:
- trajectory-based evaluation
- reward shaping across steps
- deterministic replay

---

## Observation Space

Each observation contains:

- request (natural language)
- structured availability
- constraints
- current step index
- feedback signal
- action history
- next expected stage
- reward
- done flag

Observations are designed to balance:
- realism (semi-structured inputs)
- controllability (no external dependencies)

---

## Action Space

Typed via Pydantic models:

Fields include:
- `stage`
- `proposed_time_slot`
- `confirm_schedule`
- `final_note`

Actions are structured but flexible enough to simulate agent reasoning.

---

## Reward Function

Shaped reward encourages incremental progress:

- + correct interpretation of request
- + correct use of availability constraints
- + valid slot selection
- + correct final confirmation
- + concise and relevant final note

Penalties:
- invalid stage transitions
- incorrect slot selection
- repeated or redundant actions

Properties:
- dense (not sparse)
- deterministic
- aligned with task completion

---

## Determinism & Reproducibility

- No randomness in dataset or transitions
- Fixed scenario ordering
- Identical rewards for identical actions
- Deterministic baseline policy

This ensures:
- reproducible scoring
- stable evaluation across runs
- compatibility with automated grading

---

## Baseline (Inference)

A deterministic baseline is provided.

Characteristics:
- follows correct stage sequence
- selects known valid slot
- produces consistent output
- no external model dependency

### Required Output Format

The script emits strictly formatted logs:

[START] task=<task_name> env=<env_name> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>


This format is required for evaluation pipelines.

---

## Validation & Testing

The environment has been verified with:

- `uv run openenv validate .`
- deterministic baseline execution
- pytest suite covering:
  - environment flow
  - state transitions
  - reward correctness
  - inference execution
  - API health

All tests pass from repository root.

---

## Deployment

### Docker

```bash
docker build -t smart-calendar-env .
docker run -p 8000:8000 smart-calendar-env

```
Health check:

curl http://localhost:8000/health

Expected:

{"status":"healthy"}
Hugging Face Spaces
Deploy using Docker SDK
Use repository root as build context
Verify /health endpoint
Ensure logs show clean startup

Key Design Decisions
Stage-based decomposition → improves interpretability and grading
Small synthetic dataset → ensures determinism and fast validation
Structured actions → enables consistent evaluation
Shaped rewards → provides meaningful learning signal
Root-level Dockerfile → simplifies deployment pipeline
Evaluation Alignment

This environment directly satisfies OpenEnv requirements:

real-world task simulation
multi-step agent interaction
deterministic graders
meaningful reward shaping
reproducible baseline
Docker + HF Spaces deployability
Summary

Smart Calendar Resolver is a compact, deterministic environment that captures a realistic scheduling workflow while remaining easy to validate, deploy, and evaluate.

It is designed to test:

multi-step reasoning
constraint handling
structured decision making
trajectory-based agent performance

I also pushed this to huggingface spaces
