// sokoban-solver: a push-optimal Sokoban solver for hallway-and-rooms levels.
//
// The levels this program answers are built as one hallway with small rooms off it; a box pushed
// out of a room into the hallway can never come back, so each room is an independent problem and
// the level's push optimum is the sum of its rooms'. The solver therefore works in three layers:
//
//   1. Split the level into rooms: the dead squares of the player's row (the hallway: a box there
//      reaches no goal) are cut out of the floor, and each part left holding boxes is a room.
//   2. Solve each room push-optimally with an A* search over pushes (never over steps): a state is
//      the boxes' squares plus the player's reachable region, the cost of an edge is one push, and
//      the estimate is the cheapest assignment of boxes to goals under precomputed one-box push
//      distances — admissible, so the first solution found is push-optimal. Two deadlock prunes
//      keep the search small: a box pushed onto a dead square, and a frozen 2x2 block of walls and
//      boxes with a box off its goal.
//   3. Stitch the rooms together with shortest walks on the full board and replay the whole answer
//      through a built-in simulator before writing it; a level that will not split falls back to
//      the same push search over the whole board, then to a greedier weighting of it.
//
// Every budget below is a count of search states, never the clock, and every tie breaks by square
// order, so two runs of this program on the same input write byte-identical answers.
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** States one room search may generate before it gives up (the rooms here need far fewer). */
const ROOM_STATES = 350_000;
/** States a whole-board fallback search may generate before it gives up. */
const BOARD_STATES = 700_000;

/** The four directions: [dx, dy, the step's letter, the push's letter]. Fixed order: determinism. */
const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

/**
 * A level in XSB text (# wall, space floor, . goal, $ box, * box on a goal, @ player, + player on
 * a goal): its walls (short rows and the outermost frame padded as walls), goals, boxes, player.
 */
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      if (ch === '#') continue;
      const i = y * w + x;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let x = 0; x < w; x++) {
    wall[x] = 1;
    wall[(h - 1) * w + x] = 1;
  }
  for (let y = 0; y < h; y++) {
    wall[y * w] = 1;
    wall[y * w + w - 1] = 1;
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * For each goal, the fewest pushes that move one box from any square onto it, the other boxes
 * ignored: a breadth-first walk backwards from the goal over pulls (a box on x came from x - d
 * only when the square behind it, x - 2d, was open for the player to stand on). Infinity where a
 * box can never arrive; a square is "live" when some goal's table reaches it, dead otherwise.
 */
function pushDistances(level) {
  const { w, wall, goals } = level;
  const size = wall.length;
  return goals.map((g) => {
    const dist = new Float64Array(size).fill(Infinity);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const y = x - d;
        if (y < 0 || y >= size) continue;
        if (!wall[y] && !wall[y - d] && dist[y] === Infinity) {
          dist[y] = dist[x] + 1;
          queue.push(y);
        }
      }
    }
    return dist;
  });
}

/** The squares some goal's push-distance table reaches: a box anywhere else is lost for good. */
function liveSquares(level, dists) {
  const live = new Uint8Array(level.wall.length);
  for (let s = 0; s < live.length; s++) {
    for (const d of dists) {
      if (d[s] < Infinity) {
        live[s] = 1;
        break;
      }
    }
  }
  return live;
}

/**
 * An admissible estimate of the pushes a box position still needs: the cheapest assignment of
 * boxes to distinct goals under the one-box push distances (a relaxation — boxes there pass
 * through each other — so it never overstates). Small cases run a DP over goal subsets; wide
 * ones fall back to the sum of each box's nearest-goal distance, still admissible. Infinity when
 * no assignment exists, which makes the position a deadlock however the boxes move.
 */
function estimate(boxes, dists) {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 8 || dists.length > 10) {
    let sum = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (const d of dists) if (d[b] < m) m = d[b];
      if (m === Infinity) return Infinity;
      sum += m;
    }
    return sum;
  }
  const m = dists.length;
  const dp = new Float64Array(1 << m).fill(Infinity);
  dp[0] = 0;
  for (let mask = 0; mask < dp.length; mask++) {
    let k = 0;
    for (let x = mask; x; x &= x - 1) k++;
    if (k >= n || dp[mask] === Infinity) continue;
    for (let g = 0; g < m; g++) {
      if (mask & (1 << g)) continue;
      const c = dists[g][boxes[k]];
      if (c === Infinity) continue;
      const next = mask | (1 << g);
      if (dp[mask] + c < dp[next]) dp[next] = dp[mask] + c;
    }
  }
  let best = Infinity;
  for (let mask = 0; mask < dp.length; mask++) {
    let k = 0;
    for (let x = mask; x; x &= x - 1) k++;
    if (k === n && dp[mask] < best) best = dp[mask];
  }
  return best;
}

/**
 * Flood the squares the player reaches from `from` around walls and the boxes in `occupied`,
 * stamping them in `seen` with `stamp`; returns the region's lowest square, its canonical name.
 */
function flood(level, occupied, from, seen, stamp, stack) {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    const up = p - w;
    const dn = p + w;
    const lf = p - 1;
    const rt = p + 1;
    if (!wall[up] && !occupied[up] && seen[up] !== stamp) {
      seen[up] = stamp;
      stack[n++] = up;
    }
    if (!wall[dn] && !occupied[dn] && seen[dn] !== stamp) {
      seen[dn] = stamp;
      stack[n++] = dn;
    }
    if (!wall[lf] && !occupied[lf] && seen[lf] !== stamp) {
      seen[lf] = stamp;
      stack[n++] = lf;
    }
    if (!wall[rt] && !occupied[rt] && seen[rt] !== stamp) {
      seen[rt] = stamp;
      stack[n++] = rt;
    }
  }
  return low;
}

/**
 * Whether the box just pushed onto `t` sits in a frozen 2x2 block: four squares of walls and
 * boxes with at least one box off its goal. Such a block never moves again, so it is a deadlock.
 */
function frozenBlock(level, occupied, goalSet, t) {
  const { w, wall } = level;
  const size = wall.length;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= size) continue;
    const solid = (s) => wall[s] === 1 || occupied[s] === 1;
    if (!solid(a) || !solid(b) || !solid(c) || !solid(d)) continue;
    const stranded = (s) => occupied[s] === 1 && goalSet[s] === 0;
    if (stranded(a) || stranded(b) || stranded(c) || stranded(d)) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in step letters; null if none. */
function walkPath(level, occupied, from, to) {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/**
 * A* over pushes on one board (a room, or a whole level as a fallback). A state is the boxes'
 * squares and the player's reachable region, named by the region's lowest square; an edge is one
 * push and costs 1. With weight 1 the estimate is admissible and consistent, so the first goal
 * state popped is push-optimal; a larger weight leans greedy and settles for any solution.
 * Returns {pushes, moves} (steps lower-case, pushes upper-case) or null past `maxStates` states.
 */
function solvePushes(level, maxStates, weight) {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const dists = pushDistances(level);
  const live = liveSquares(level, dists);
  if (level.boxes.some((b) => !live[b])) return null;
  const solved = (boxes) => boxes.every((b) => goalSet[b] === 1);
  if (solved(level.boxes)) return { pushes: 0, moves: '' };
  const h0 = estimate(level.boxes, dists);
  if (h0 === Infinity) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size); // the region of the state being expanded
  const mark = new Int32Array(size); // the region of a child state, kept apart from `seen`
  const stack = new Int32Array(size);
  let stamp = 0;
  const keyOf = (boxes, low) => `${low}|${boxes.join(',')}`;

  // The state store: boxes, the player's square (where the last push left it), parentage, and
  // the push that made the state, so the answer's walks can be rebuilt at the end.
  const sBoxes = [level.boxes.slice()];
  const sPlayer = [level.player];
  const sParent = [-1];
  const sFrom = [-1];
  const sTo = [-1];
  const sG = [0];
  const sKey = [''];
  const best = new Map();
  for (const b of level.boxes) occupied[b] = 1;
  const low0 = flood(level, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  sKey[0] = keyOf(level.boxes, low0);
  best.set(sKey[0], 0);

  // The open list: a binary heap on (f, deeper first, insertion order) — all fixed, so the
  // search unfolds identically on every run.
  const heap = [];
  let counter = 0;
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e) => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  heapPush([weight * h0, 0, counter++, 0]);

  let found = -1;
  while (heap.length > 0 && found < 0) {
    const s = heapPop()[3];
    if (sG[s] !== best.get(sKey[s])) continue; // a cheaper route to this state superseded it
    const boxes = sBoxes[s];
    if (solved(boxes)) {
      found = s;
      break;
    }
    const g = sG[s];
    for (const b of boxes) occupied[b] = 1;
    flood(level, occupied, sPlayer[s], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const stand = b - d;
        const t = b + d;
        if (seen[stand] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goalSet, t);
        let child = null;
        let childKey = null;
        if (!frozen) {
          child = boxes.slice();
          child[bi] = t;
          child.sort((x, y) => x - y);
          childKey = keyOf(child, flood(level, occupied, b, mark, ++stamp, stack));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen) continue;
        const g2 = g + 1;
        const known = best.get(childKey);
        if (known !== undefined && known <= g2) continue;
        const h = estimate(child, dists);
        if (h === Infinity) continue;
        if (sBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        best.set(childKey, g2);
        const idx = sBoxes.length;
        sBoxes.push(child);
        sPlayer.push(b);
        sParent.push(s);
        sFrom.push(b);
        sTo.push(t);
        sG.push(g2);
        sKey.push(childKey);
        heapPush([g2 + weight * h, -g2, counter++, idx]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Rebuild the answer: walk to each push's stance, then push, state by state from the start.
  const chain = [];
  for (let s = found; s > 0; s = sParent[s]) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  const occ = new Uint8Array(size);
  for (const s of chain) {
    const d = sTo[s] - sFrom[s];
    for (const b of boxes) occ[b] = 1;
    const walk = walkPath(level, occ, player, sFrom[s] - d);
    for (const b of boxes) occ[b] = 0;
    if (walk === null) return null;
    moves += walk + DIRS.find(([dx, dy]) => dy * w + dx === d)[3];
    player = sFrom[s];
    boxes = sBoxes[s];
  }
  return { pushes: chain.length, moves };
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway) out of the floor;
 * each part left is a room with its squares, boxes and goals. Null when the level is no such
 * set of rooms — a box in the hallway, a goal outside every part, or a part whose boxes and
 * goals do not balance — and the caller falls back to a whole-board search.
 */
function decompose(level) {
  const { w, wall, goals, boxes, player } = level;
  const size = wall.length;
  const live = liveSquares(level, pushDistances(level));
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (q < 0 || q >= size) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) {
    if (part[b] < 0) return null;
    rooms[part[b]].boxes.push(b);
  }
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Replay `moves` from a position: the position it reaches, or null on an illegal step. */
function simulate(level, boxes, player, moves) {
  const { w, wall } = level;
  const boxSet = new Set(boxes);
  let p = player;
  let pushes = 0;
  for (const ch of moves) {
    const dir = DIRS.find(([, , letter]) => letter === ch.toLowerCase());
    if (!dir) return null;
    const d = dir[1] * w + dir[0];
    const q = p + d;
    if (wall[q]) return null;
    if (boxSet.has(q)) {
      const t = q + d;
      if (wall[t] || boxSet.has(t)) return null;
      boxSet.delete(q);
      boxSet.add(t);
      pushes++;
    }
    p = q;
  }
  return { boxes: [...boxSet].sort((a, b) => a - b), player: p, pushes };
}

/**
 * Solve a level room by room: each room on its own board — its squares plus the one hallway
 * square outside its door, where the player starts — then the player walks on across the full
 * board to the next room. Returns {pushes, moves} or null when any step fails.
 */
function solveByRooms(level) {
  const rooms = decompose(level);
  if (rooms === null) return null;
  const { w, wall, player: startPlayer } = level;
  const size = wall.length;
  const row = Math.floor(startPlayer / w);
  const occupied = new Uint8Array(size);
  let boxes = level.boxes.slice();
  let player = startPlayer;
  let moves = '';
  let pushes = 0;
  for (const room of rooms) {
    // The porch: the hallway square outside this room's door — the room's only way in.
    let porch = -1;
    for (const s of room.squares) {
      for (const [dx, dy] of DIRS) {
        const q = s + dy * w + dx;
        if (q >= 0 && q < size && Math.floor(q / w) === row && !wall[q]) {
          if (porch >= 0 && porch !== q) return null;
          porch = q;
        }
      }
    }
    if (porch < 0) return null;
    const subWall = new Uint8Array(size).fill(1);
    for (const s of room.squares) subWall[s] = 0;
    subWall[porch] = 0;
    const sub = {
      w,
      h: level.h,
      wall: subWall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player: porch,
    };
    const solvedRoom = solvePushes(sub, ROOM_STATES, 1);
    if (solvedRoom === null) return null;
    for (const b of boxes) occupied[b] = 1;
    const walk = walkPath(level, occupied, player, porch);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    const segment = walk + solvedRoom.moves;
    const after = simulate(level, boxes, player, segment);
    if (after === null) return null;
    boxes = after.boxes;
    player = after.player;
    moves += segment;
    pushes += solvedRoom.pushes;
  }
  const goalSet = new Set(level.goals);
  if (!boxes.every((b) => goalSet.has(b))) return null;
  return { pushes, moves };
}

/** One level's answer: its moves in LURD, or '' when nothing the solver tries answers it. */
function solveLevel(text) {
  const level = parseLevel(text);
  if (level.boxes.length === 0 || level.player < 0) return '';
  const goalSet = new Set(level.goals);
  if (level.boxes.every((b) => goalSet.has(b))) return '';
  const verify = (moves) => {
    const end = simulate(level, level.boxes, level.player, moves);
    return end !== null && end.boxes.every((b) => goalSet.has(b));
  };
  const byRooms = solveByRooms(level);
  if (byRooms !== null && verify(byRooms.moves)) return byRooms.moves;
  const whole = solvePushes(level, BOARD_STATES, 1);
  if (whole !== null && verify(whole.moves)) return whole.moves;
  const greedy = solvePushes(level, BOARD_STATES, 3);
  if (greedy !== null && verify(greedy.moves)) return greedy.moves;
  return '';
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const levelFile = join(INPUT_DIR, id, 'level.txt');
  if (!existsSync(levelFile)) continue;
  const moves = solveLevel(readFileSync(levelFile, 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
