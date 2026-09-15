<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LH Task Encoding Rules — Aligning Subtask Shape with the GT Monitor

How long-horizon tasks in robolab should be encoded so the CSM
(`ConditionalsStateMachine`) emits the per-object signals that the
aspect-1 task-failure logger consumes.

This is the **task-side prerequisite** for aspect 1.  The actual
per-task edits are manual — this doc captures only the rules and the
why; the per-task encoding decisions involve task-specific judgment
the author makes when authoring or revising each file.

For aspect-1 implementation details (event taxonomy, rule pipeline,
robolab exporter changes), see `../plans/aspect1-implementation-plan.md`.

---

## Background: the two state machines

| Layer | File | Behavior | Encodes |
|---|---|---|---|
| `SubtaskStateMachine` (outer) | `robolab/core/task/subtask_state_machine.py` | **Strictly sequential.**  Instantiates one `ConditionalsStateMachine` at a time; advances only when current returns `complete=True`.  Subtask weights normalize over `Σ subtask.score`. | **Order between phases.** |
| `ConditionalsStateMachine` (inner) | `robolab/core/task/conditionals_state_machine.py` | **Parallel** per-object ladders.  `total_score` aggregates per `logical` mode (`all`=mean, `any`=max, `choose`=top-k).  Per-object regression-aware. | **Parallelism within a phase.** |

The implication is the rule below.

---

## The rule

> **One `Subtask` per ordered phase.  Within each `Subtask`, list every
> object that should be done in any order during that phase.**

> **A "phase" is an ordered constraint imposed by the *success contract*,
> not by the most convenient policy strategy.**  If the success
> terminator only checks where objects end up, the whole task is one
> phase regardless of how many intermediate moves a typical policy
> would make.

Mechanically:

- A task with N strictly-ordered phases (e.g. unstack-then-restack
  where stack physics actually forces order) → `subtasks = [Subtask, …]`
  of length N.  Use multi-subtask encoding **only** when the success
  contract genuinely requires an ordered intermediate state.
- A task whose objects can be done in any order, even with multiple
  destinations → exactly **one** `Subtask` containing all objects.
  Use `pick_and_place(container=…)` for shared destination,
  `pick_and_place_on_surface(surface=…)` for surface-targeted, or
  `pick_and_place_grouped(groups=[…])` for multiple destinations.
  The policy is free to use a buffer, interleave, or take any order;
  per-object completion tracks final-state progress regardless of path.

A common antipattern this rules out: encoding a swap or restore task
as `[buffer-A-on-table, place-B, place-A-in-final-spot]` because
that's the most convenient maneuver.  The buffer step is policy
strategy, not success contract — bake it into the score and you
penalize policies that achieve the same end-state by a different path,
while masking real progress as flat or zero score until the encoded
subtask aligns.

### Decision table

| Task shape | Subtasks | Composite |
|---|---|---|
| All objects → one container, any order | 1 | `pick_and_place(object=[…], container=X)` |
| All objects → one surface, any order | 1 | `pick_and_place_on_surface(object=[…], surface=X)` |
| Different objects → different containers, any order | 1 | `pick_and_place_grouped(groups=[…])` |
| Strict order (A first, B second, …) | N | One composite per phase |
| Hybrid: ordered phases each with internal any-order | N | Per-phase composite |
| Spatial / arrangement success (`object_left_of`, `object_in_front_of`, `stacked`) | 1+ | `Subtask(partial(<predicate>, object=[…]))` — see "Atomic compound subtasks" below |

### Why this works for the GT monitor

| Aspect-1 event | Behavior under the rule |
|---|---|
| OBJECT_COMPLETE | Fires per-object as each one reaches its destination, even with multiple destinations.  For ordered phases, advances phase-by-phase as each `Subtask` completes. |
| WRONG_TARGET_PLACE | Per-object `condition_idx 2` (released) and `3` (in_target) exposed for every active-phase target.  Fires when an object is in a stable released-and-not-in-target state. |
| WRONG_OBJECT_PICKED | Active-phase targets are visible to the rule via `current_subtask_targets`; off-target grasps fire correctly. |
| OBJECT_REGRESSION | Once an object emits OBJECT_COMPLETE, its `object_completed` flag is watched cross-subtask.  Regression fires if the agent later moves it off-target. |
| STUCK | `subtask.score` increases as soon as any active-phase object progresses; STUCK fires only when truly nothing moves. |

---

## Composites available

All in `robolab/core/task/conditionals.py`:

```python
pick_and_place(object=[…], container="bin", logical="all")
# Per-object 4-rung ladder: grabbed → above_bottom(bin) → dropped → in_container(bin)

pick_and_place_on_surface(object=[…], surface="table", logical="all")
# Per-object 4-rung ladder: grabbed → above_bottom(surface) → dropped → on_top(surface)

pick_and_place_grouped(groups=[
    {"object": [...], "container": "bin_a"},
    {"object": [...], "container": "bowl"},
], logical="all")
# Multi-destination any-order: each object's 4-rung ladder
# advances to its own group's container.
```

For atomic-compound success predicates (e.g. `stacked`,
`object_left_of`, `object_in_front_of`), use `Subtask(conditions=
partial(<predicate>, object=[<targets>]))` — see next section.

---

## Atomic compound subtasks

When the success terminal is a multi-object spatial predicate
(`stacked([a, b, c])`, `object_left_of(a, b)`, etc.) rather than a
per-object placement, the natural encoding is a single-condition
`Subtask`:

```python
Subtask(
    name="restack",
    conditions=partial(
        stacked,
        objects=["red_block", "blue_block", "green_block"],
        order="bottom_to_top",
        # The exporter reads `object=` or `objects=` from the partial's
        # keywords to expose real scene-object names instead of the
        # placeholder "conditions" CSM group key.
        object=["red_block", "blue_block", "green_block"],
    ),
    score=1.0,
)
```

The `object=` (or `objects=`) kwarg on the partial is what the
exporter (`_export_conditions`, `_export_object_completed`) reads to
populate `gt_state.subtask.conditions` and `object_completed` with the
real scene-object names.  Without it, all the rules see is the generic
`"conditions"` placeholder key and per-object detection breaks.

**The exporter handles this fallback automatically as of the recent
fix** — atomic compound subtasks now expose real names regardless.
But the convention "always pass `object=[…]` on compound success
partials" is still recommended for clarity.

For tasks where the spatial relation is symmetric (e.g. swap by
position), use a **dict-form Subtask with both objects as group keys**
so both are exposed as targets independently:

```python
Subtask(
    name="swap_block_positions",
    conditions={
        "red_block":  partial(object_left_of,  object="red_block",  reference_object="blue_block"),
        "blue_block": partial(object_right_of, object="blue_block", reference_object="red_block"),
    },
    logical="all",
    score=1.0,
)
```

This avoids the "single object as target" pitfall where the rule
would false-fire WRONG_OBJECT_PICKED for the other-side block being
moved.

---

## Per-task-type guidance (high-level only)

The 50+ LH tasks broadly fall into the following shapes.  Authoring
each task is a manual judgment call; this section captures the
default rule per shape.

| Shape | Default rule |
|---|---|
| **Recall — final-state-only** | One subtask matching the success terminator (`pick_and_place(...)`).  Don't encode the choreographic "place on table first" step unless the success terminator literally requires it. |
| **Recall — empty-then-place (compound success)** | Two ordered subtasks: phase 1 = `pick_and_place_on_surface(items, "table")` if the terminator requires "items not in source"; phase 2 = `pick_and_place(...)` for the destination. |
| **Recall — unstack-then-place** | Sequential: per-block `pick_and_place_on_surface([<block>], "table")` for unstack phases (top-down by stack physics), then the final placement composite.  Each unstack subtask is independent so the policy can choose any per-block strategy. |
| **Swap (container-to-container)** | One grouped subtask: `pick_and_place_grouped(groups=[<each item>→<its swapped destination>])`.  The buffer move is policy strategy; don't encode it. |
| **Swap (positional)** | Single `Subtask` with **all swapped objects as group keys** (each with the appropriate spatial predicate as its single condition), `logical="all"`. |
| **Reverse / Restack tower** | Sequential: top-down unstack phases (each `pick_and_place_on_surface`), then bottom-up restack phases each as `Subtask(partial(stacked, objects=[<cumulative>]))`.  N-block tower → N unstack subtasks + N restack subtasks. |
| **LH_A unstack-by-position** | Same as Reverse: top-down unstack to table, then `pick_and_place(<the-target-cube>, <bin/bowl>)`. |
| **Cycle (container)** | Same as Swap: one `pick_and_place_grouped`. |
| **Rotate (positional 4-corner)** | Single `Subtask(partial(<arrangement_predicate>, object=[<all 4>]))` if the success is an atomic arrangement check.  More granular per-position encoding is possible but is per-task design work. |
| **Misc spatial (line / front / etc.)** | Same as positional swap — dict-form Subtask with each object as a group key. |

The general scoring convention: weight each pick-place op equally
across the task (e.g. 1/N for an N-step task), not front-loaded.

---

## Cross-cutting things to watch for

A few patterns that previously caused silent breakage; the rules above
already preclude them, but called out for awareness:

1. **Compound-success partials without `object=` kwarg.**  The
   exporter previously fell back to a placeholder `"conditions"` key
   for `object_completed` when a Subtask used an atomic compound
   predicate without keywords.  The recent exporter fix
   (`_export_conditions`) reads `partial.keywords["objects"|"object"]`
   automatically — but passing `object=[…]` explicitly is still
   recommended for code clarity.

2. **`pick_and_place(container="table")`.**  `object_in_container(table)`
   is never satisfied.  Use `pick_and_place_on_surface(surface="table")`
   if buffering on the table is the actual phase.

3. **Inline `[partial(grabbed, X), partial(outside_of, X)]` lists** as
   a Subtask's `conditions`.  These sanitize to `{"group1": …,
   "group2": …}` (multiple single-condition groups instead of one
   multi-rung ladder), losing per-object cond_idx 2/3 exposure.
   `WRONG_TARGET_PLACE` becomes silent for those phases.  Use the
   composites instead, or a dict-form `Subtask` with one object → one
   condition list.

4. **`pick_and_place_grouped` only takes containers.**  For mixed
   destinations (some objects → table, others → bin), use multiple
   subtasks or a manual dict-form `Subtask` with per-object ladders.

---

## What this doc does NOT cover

- **VLM-introduced intermediate subgoals.**  When the orchestrator's
  VLM planner decomposes a task into intermediate steps that don't
  align with CSM-tracked completions (e.g. "pick up X" without a
  destination), aspect-2's per-subgoal monitor has a granularity
  problem.  Aspect 1 doesn't share this problem because it's
  task-scoped; aspect 2's planner-mapping cleanup is a separate
  workstream.

- **Per-task encoding edits.**  These are manual.  The rules above
  define the discipline; applying them to each task involves
  understanding what the success contract actually requires,
  identifying which physics steps are forced vs which are policy
  strategy, and choosing the right composite.  See git history of
  `robolab/tasks/robovolo/` for the applied patterns.

- **Policy weakness vs detector weakness separation.**  An encoded
  task can be correct but the policy can't do it.  Use a passthrough
  baseline before attributing eval lift to encoding changes.
