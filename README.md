# sokoban-solver-01-run-20261121

A push-optimal Sokoban solver, built for make.fun's
[sokoban-solver-01-run-20261121](https://make.fun/bounty/sokoban-solver-01-run-20261121) bounty.

`main.ts` (Node 22, no dependencies) reads every instance under `/input/<id>/level.txt`
(XSB format) and writes the player's moves to `/output/<id>/solution.txt` in LURD
(lower-case steps, upper-case pushes). The local directories can be overridden with
`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`, as in the kit's starter.

## How it works

The levels are built as one hallway with small rooms off it, and a box pushed out of a
room into the hallway can never come back — so each room is an independent problem and a
level's push optimum is the sum of its rooms'. The solver mirrors that structure:

1. **Split** — the dead squares of the player's row (the hallway: a box there reaches no
   goal) are cut out of the floor; each part left holding boxes is a room.
2. **Solve each room push-optimally** — an A\* search over *pushes*, not steps. A state is
   the boxes' squares plus the player's reachable region (named by its lowest square).
   The heuristic is the cheapest assignment of boxes to distinct goals under precomputed
   one-box push distances (reverse-pull BFS tables) — admissible and consistent, so the
   first solution found is push-optimal. Dead-square and frozen-2×2 deadlock pruning keep
   the search small.
3. **Stitch and verify** — rooms are joined by shortest walks on the full board, and the
   whole answer is replayed through a built-in simulator before anything is written.

A level that will not split falls back to the same push search over the whole board, then
to a greedier weighting of it, so any solvable level still gets an answer.

Every budget is a count of search states, never the clock, and every tie breaks by square
order: two runs on the same input write byte-identical answers.

## Results

On the kit's practice set the solver matches the known push optimum P\* on all five
instances (13 / 16 / 29 / 47 / 60 pushes) — a practice score of 1,000,000 under the
competition's checker, against the house plain-A\* baseline's 200,000 — and finishes the
whole set in about 0.2 s of the 60 s wall.

## Layout

- `main.ts` — the whole solver (parser, push-distance tables, A\* push search, room
  decomposition, stitching, self-verification, driver).
- `.makefun/owner.json` — the entry's claim file.
- `LICENSE` — MIT.
