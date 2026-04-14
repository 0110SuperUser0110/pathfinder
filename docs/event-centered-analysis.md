# Event-Centered Analysis

Pathfinder analyzes labeled experiences as event-centered timelines over the full EEG recording.

## Core Rule

Do not hardcode any specific state name into the code. Labels such as `happy`, `calm`, `pain_high`, `visual_stimulus_present`, or `sexual_arousal` should only appear as values in metadata.

Use generic fields instead:

- `event_family`
- `target_label`
- `event_subtype`
- `label_namespace`

## Why

A target pattern may live in:

- the lead-in before the event
- the onset transition
- the sustained interval
- the offset transition
- the recovery interval after the event

If Pathfinder only analyzes isolated, context-free windows, it can miss the pattern entirely.

## Correct Processing Model

1. Keep the full recording unchanged.
2. Mark events on the timeline.
3. Extract event-centered context windows.
4. Preserve those event-centered windows as EEG artifacts.
5. Run both local-window and sequence-level analysis later.
6. Tie every derived artifact back to the original recording and time span.

## Phase 1 Event Windows

Pathfinder now supports configurable windows for:

- `pre_event`
- `onset`
- `sustained`
- `offset`
- `post_event`

Optional baseline windows can also be recorded as metadata-backed signal segments without forcing subtraction by default.

## Anti-Pattern

Do not write code like:

- `if label == "happy": ...`
- `if arousal_type == "sexual": ...`
- model branches specialized around one named state

## Correct Pattern

Write code that does this instead:

- accept a generic event label
- preserve the full recording reference
- derive windows from timeline metadata
- run the same extraction path for every label
- compare labels only at the analysis layer
- keep transformed branches separate from raw-preserving branches
