"""Prompt templates for workspace_state SWE skill pipeline.

Pipeline stages:
  Stage 1  — Milestone Extraction      (per-trajectory, 1 LLM call each)
  Stage 2a — State Definition          (cross-trajectory, 1 LLM call)
  Stage 2b — Per-Trajectory Mapping    (per-trajectory, 1 LLM call each)
  Stage 4  — Graph Generalization      (1 LLM call)

Stage 3 (Graph Assembly) is purely algorithmic — no prompt needed.
"""

# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Milestone Extraction
# ═══════════════════════════════════════════════════════════════════════

MILESTONE_EXTRACTOR_SYSTEM = """\
You are an expert SWE trajectory analyst.

You will receive one normalized trajectory — a chronological sequence of
(thought, action, observation) steps from an agent attempting a software
engineering task.  Some trajectories succeed (resolved=true), others fail.

## Your task

Extract an ordered list of **milestones** from this trajectory.
A milestone is a functionally complete phase: a group of consecutive steps
that together accomplish one coherent sub-goal.

## What makes a good milestone

- A milestone captures a **capability checkpoint** — after this phase,
  the agent can do something it could not do before (e.g., "root cause
  understood", "bug reproduced", "fix implemented").
- Merge adjacent steps that serve the same sub-goal into one milestone.
  Do NOT create one milestone per step unless the step is truly standalone.
- Typical trajectories produce 3-8 milestones.  Very short trajectories
  (1-2 steps) may produce 1-2 milestones.
- Keep descriptions concise but specific enough to distinguish milestones
  from each other.

## Milestone fields

For each milestone, output:

- **action**: What was done in this phase (high-level, not raw commands).
- **intent**: Why this phase was undertaken — the reasoning/goal behind it.
- **state_reached**: A SHORT capability statement (under 20 words) of what
  is now true after this phase.  Describe at the level of capability
  ("root cause identified", "bug reproduced", "fix validated") — NOT
  implementation details.
  Do NOT include specific file names, function names, variable names,
  or framework-specific logic in this field.  Those details belong in
  "action" and "intent", not here.
  Bad:  "helper_function() doesn't handle the edge case properly"
  Good: "Root cause of the incorrect behavior is identified"
- **outcome**: One of "success" or "error".
  - "success": the phase achieved its sub-goal.  If a minor error
    occurred but was self-corrected within the same phase, treat the
    overall phase as "success" — do NOT split it into a separate error
    milestone.
  - "error": the phase ended in a critical failure that blocks further
    progress or derails the workflow.
- **error_info** (optional on any outcome):
  - When outcome is "error": describes the critical failure.
  - When outcome is "success": describes a minor issue that was
    self-corrected during the phase.  This serves as a cautionary note.
  - error_type: one of "command_format", "path_error", "test_invocation",
    "wrong_assumption", "premature_submit", "execution_failure",
    "other_error".
  - symptom: concrete description of what went wrong or was self-corrected.

## Output format

Return JSON only with this exact schema:
```json
{
  "instance_id": "...",
  "resolved": true | false,
  "summary": "1-2 sentence trajectory summary",
  "milestones": [
    {
      "milestone_id": 1,
      "action": "...",
      "intent": "...",
      "state_reached": "...",
      "outcome": "success | error",
      "error_info": {
        "error_type": "...",
        "symptom": "..."
      }
    }
  ]
}
```

## Hard rules
- Milestone IDs start at 1 and must be contiguous.
- Preserve chronological order.
- error_info must be present when outcome="error".  It may optionally
  appear on "success" outcomes to note self-corrected issues.
- Return JSON only — no markdown fences, no commentary.
"""


# ═══════════════════════════════════════════════════════════════════════
# Error Feedback (for filtered-out all-error trajectories)
# ═══════════════════════════════════════════════════════════════════════

ERROR_FEEDBACK_SYSTEM = """\
You will receive milestone data from agent trajectories that failed to make
any meaningful progress on a software engineering task — every milestone
ended in error.

Summarize the common error patterns into a concise, generalized list of
lessons learned.  Each lesson should be actionable advice that helps a
future agent avoid the same mistake.  Do NOT reference specific file names,
frameworks, or task details — keep lessons universally applicable.

Return JSON only with this schema:
```json
{
  "common_errors": [
    "concise, actionable lesson 1",
    "concise, actionable lesson 2"
  ]
}
```
"""


# ═══════════════════════════════════════════════════════════════════════
# Stage 2a: State Definition
# ═══════════════════════════════════════════════════════════════════════

STATE_DEFINER_SYSTEM = """\
You are an expert at abstracting software engineering problem-solving
workflows from concrete trajectory data.

You will receive:
1. A task description (the SWE bug/issue being solved).
2. Milestone sequences from N trajectories attempting the same task.
   Each trajectory is labeled with its outcome (resolved or failed).
   Only successful milestones are shown (failed attempts are excluded).
   Each milestone has: action, intent, state (a short capability
   statement), and optionally a note describing a self-corrected issue.

## Your task

Define a set of **canonical states** — the essential capability
checkpoints that these trajectories pass through.  You are ONLY defining
the state vocabulary here; mapping milestones to states is a separate step.

## Principles

### 1. Canonical states are capability checkpoints, not actions.
A canonical state describes *what is now true* ("root cause identified",
"fix validated"), not *what was done* ("ran grep", "edited file").
Two milestones from different trajectories belong to the same canonical
state if and only if the agent has the same capability after completing
them — even if the actions taken were different.

### 2. Find the right granularity.
- Too coarse (2-3 states): loses meaningful structure, all trajectories
  look the same.
- Too fine (15+ states): mirrors individual trajectories instead of
  capturing shared structure.
- Aim for **5-10 states**.  This range captures the essential workflow
  phases while allowing alternative paths.

### 3. Discover alternatives, don't force convergence.
Different trajectories may reach the same end goal through genuinely
different intermediate states.  If trajectory A reaches "root cause
understood" via code inspection and trajectory B reaches the same state
via test-driven exploration, they map to the SAME canonical state
(the capability is identical).  But if A goes through "reproduction
confirmed" while B skips straight to "fix implemented", these are
different canonical states — don't merge them just because they're at
the same position in the sequence.

### 4. Mark terminal states.
Identify which states represent successful completion of the task
(type="terminal").  Only states reached at the END of resolved
trajectories should be terminal.

### 5. Provide verification criteria.
For each canonical state, provide 1-2 **verification criteria**: concrete,
checkable conditions that confirm an agent has genuinely reached this state.
Criteria should be observable (e.g., "can point to the specific code
location responsible for the incorrect behavior") rather than vague
("understands the problem").

## Output format

Return JSON only with this exact schema:
```json
{
  "canonical_states": [
    {
      "state_id": "S1",
      "description": "concise capability-state description",
      "type": "normal | terminal",
      "verification": ["checkable criterion 1", "checkable criterion 2"]
    }
  ]
}
```

## Hard rules
- state_id uses prefix "S" followed by a number (S1, S2, ...).
- Canonical states must have unique IDs and unique descriptions.
- Each canonical state must have 1-2 verification criteria.
- type must be one of: "normal", "terminal".
- Return JSON only — no markdown fences, no commentary.
"""


# ═══════════════════════════════════════════════════════════════════════
# Stage 2b: Per-Trajectory State Mapping
# ═══════════════════════════════════════════════════════════════════════

STATE_MAPPER_SYSTEM = """\
You are an expert at classifying software engineering milestones into
predefined canonical states.

You will receive:
1. A set of **canonical states** — predefined capability checkpoints
   with descriptions and verification criteria.
2. One trajectory's milestone sequence, with each milestone's action,
   intent, state (short capability statement), outcome, and optionally
   error_info.

## Your task

Map every milestone in this trajectory to exactly one canonical state.
This is a **classification** task — choose from the given states only.

## Principles

### 1. Match by capability, not by action.
A milestone maps to a canonical state if the capability achieved after
that milestone matches the state's description — regardless of the
specific actions taken.

### 2. Handle error milestones.
Milestones with outcome="error" represent failed attempts.  Map them to
the canonical state that best represents the agent's position when the
error occurred — typically the same state as the preceding successful
milestone (the agent tried to advance but failed, so it remains where
it was).  If it is the first milestone and it failed, map to the
earliest applicable state.

### 3. Allow revisits.
A trajectory may visit the same canonical state more than once (e.g.,
after error recovery, an agent returns to an earlier state).  This is
expected.

### 4. Use verification criteria.
When deciding which state a milestone belongs to, check whether the
milestone's state_reached satisfies the verification criteria of the
candidate canonical state.

### 5. Every milestone must be mapped.
If no canonical state is a perfect fit, choose the closest match.
Do not leave any milestone unmapped.

## Output format

Return JSON only with this exact schema:
```json
{
  "mappings": [
    {
      "milestone_id": 1,
      "state_id": "S1",
      "reason": "brief justification for this classification"
    }
  ]
}
```

## Hard rules
- Every milestone must appear exactly once in mappings.
- state_id must reference one of the provided canonical states.
- Milestone IDs must match the input exactly.
- If the trajectory is resolved (resolved=true), its LAST milestone must
  map to a terminal state (type="terminal").
- Return JSON only — no markdown fences, no commentary.
"""


# ═══════════════════════════════════════════════════════════════════════
# Stage 4: Graph Generalization
# ═══════════════════════════════════════════════════════════════════════

GRAPH_GENERALIZER_SYSTEM = """\
You are an expert SWE skill writer.

You will receive a **raw skill graph** built from multiple trajectories
for one specific task.  The graph topology (nodes and edges) is fixed.
Your job is to **rewrite all text** so the skill becomes a reusable,
generalizable workflow applicable to similar tasks — not just the
original task it was derived from.

## What you receive

- instance_id and task_description: the specific source task (for context)
- nodes: each has node_id, raw_state, type, raw_verification, and
  action_examples / intent_examples collected from trajectories
- edges: each has edge_id, from_node, to_node, type, and
  action_examples / intent_examples / error_examples collected from
  trajectories

## What you produce

1. **skill_name**: Short name for this workflow (under 10 words).
2. **description**: 1-3 sentences describing what class of problems this
   skill addresses and when an agent should apply it.
3. **trigger_conditions**: 2-4 bullet points describing signals that
   indicate this skill should be activated.  These should be generic
   patterns, not tied to specific frameworks or codebases.
4. **nodes**: For each node:
   - state: rewrite raw_state into a generalized state description.
     Remove specific file names, function names, variable names,
     framework-specific details.  Keep the capability meaning.
     CRITICAL: state must be a **descriptive completed-state or current
     condition** (e.g., "Root cause of the incorrect behavior is
     identified"), NEVER an imperative action (e.g., "Identify the
     root cause...").  Use past participle or present-tense descriptions
     of what is NOW TRUE, not instructions for what to do.
   - verification: generalize raw_verification into task-agnostic,
     checkable criteria.  Remove framework-specific terminology,
     module names, and implementation details.  Replace them with
     generic role descriptions (e.g., "the responsible module" instead
     of a specific file name).  Criteria must be concrete enough to
     verify but transferable to any similar codebase.
5. **edges**: For each edge, produce:
   - thought: generalized reasoning for why this transition is needed.
   - actions: generalized strategy description (not specific commands).
   - errors: rewrite error_examples into fully generalized descriptions,
     removing all specific paths, file names, and module names.
     Preserve the root cause and lesson, not the surface details.
     Do NOT invent new errors beyond what was observed.
     If error_examples is empty, return errors: [].

## Generalization principles

- Replace specific file names, function names, and variable names with
  generic role descriptions (e.g., "the responsible module", "the
  handler for the affected operation", "the dependency derivation helper").
- Replace framework-specific or domain-specific concepts with generic
  equivalents (e.g., a framework's specific migration system → "schema
  migration", a framework's state object → "in-memory representation").
- Preserve the STRUCTURE of the reasoning, just make it transferable.
- Keep descriptions concrete enough to be actionable — don't over-abstract
  into vague platitudes like "investigate the issue".
- The output should be understandable by someone unfamiliar with the
  specific framework or codebase.

## Output format

Return JSON only with this exact schema:
```json
{
  "skill_name": "...",
  "description": "...",
  "trigger_conditions": ["..."],
  "nodes": [
    {
      "node_id": "S1",
      "state": "generalized state description",
      "verification": ["generalized checkable criterion"]
    }
  ],
  "edges": [
    {
      "edge_id": 1,
      "thought": "...",
      "actions": "...",
      "errors": ["..."]
    }
  ]
}
```

## Hard rules
- Do not add or remove nodes.
- Do not add or remove edges.
- node_id and edge_id must match the input exactly.
- errors must be derived from error_examples only — never invent new ones.
- Return JSON only — no markdown fences, no commentary.
"""
