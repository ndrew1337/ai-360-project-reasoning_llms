
import math
import os
import re
import hashlib
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

from tot.models import gpt


_DBG_FLOAT = os.getenv("TOT_DEBUG_FLOAT") == "1"
_float_dbg_cnt = 0

def _dbg() -> bool:
    return os.getenv("TOT_DEBUG_MCTS") == "1"


def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]

    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)

    if os.getenv("TOT_DEBUG_EVAL") == "1":
        print("\n=== EVAL PROMPT ===\n", value_prompt)
        print("=== EVAL OUTPUTS (raw repr) ===")
        for i, o in enumerate(value_outputs):
            print(i, repr(o))
        print("=== END ===\n")

    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values


def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values


def get_proposals(task, x, y):
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split("\n")
    return [y + _ + "\n" for _ in proposals if _.strip()]


def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == "standard":
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == "cot":
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f"prompt_sample {prompt_sample} not recognized")
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]


# State adapters
#   - Game24Adapter: uses "(left: ...)" parsing for transpositions
#   - GenericAdapter: works for text/crosswords (no numeric parsing)

_NUM_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

def _safe_float(tok: str) -> float:
    tok = tok.strip()
    # protect against pathological huge decimal strings burning CPU in strtod
    if len(tok) > 64:
        raise ValueError("token too long")
    return float(tok)


def _nonempty_lines(y: str) -> List[str]:
    return [ln.rstrip() for ln in y.splitlines() if ln.strip()]


def _game24_step_lines(y: str) -> List[str]:
    return [ln.strip() for ln in y.splitlines() if ln.strip() and "(left:" in ln]


def _current_numbers_str_game24(x: str, y: str) -> str:
    """
    Take last '(left: ...)' line if present, else last non-empty line,
    from y if y else x.
    """
    src = y if y else x
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    if not lines:
        return src.strip()

    last_left = None
    for ln in reversed(lines):
        if "left:" in ln:
            last_left = ln
            break
    last = last_left if last_left is not None else lines[-1]

    if "left:" in last:
        try:
            return last.split("left:", 1)[1].split(")", 1)[0].strip()
        except Exception:
            return last.strip()
    return last.strip()


def _parse_nums(nums_str: str) -> Tuple[float, ...]:
    """
    Robust numeric parse: extract numeric substrings via regex
    (avoids weird tokens like '2)' or '1.2.3' causing slow/invalid float()).
    """
    global _float_dbg_cnt
    if not nums_str:
        return tuple()

    out: List[float] = []
    for m in _NUM_RE.finditer(nums_str):
        tok = m.group(0)
        if _DBG_FLOAT:
            _float_dbg_cnt += 1
            suspicious = (
                len(tok) > 40
                or any(ch in tok for ch in "eE")
                or "nan" in tok.lower()
                or "inf" in tok.lower()
            )
            if suspicious:
                print(f"[FLOAT?] tok={tok!r} from nums_str={nums_str!r}", flush=True)
            elif _float_dbg_cnt % 200000 == 0:
                print(f"[FLOAT] parsed {_float_dbg_cnt} nums; last tok={tok!r}", flush=True)

        try:
            out.append(_safe_float(tok))
        except Exception as e:
            if _DBG_FLOAT:
                print(f"[FLOAT FAIL] tok={tok!r} err={e!r} nums_str={nums_str!r}", flush=True)
            continue

    return tuple(out)


class _Game24Adapter:
    def __init__(self, x: str):
        self.x = x

    def depth(self, y: str) -> int:
        return len(_game24_step_lines(y))

    def key(self, y: str) -> Tuple[Tuple[float, ...], int]:
        nums = _parse_nums(_current_numbers_str_game24(self.x, y))
        nums = tuple(sorted(round(v, 6) for v in nums))
        return nums, self.depth(y)

    def is_terminal(self, y: str) -> bool:
        nums = _parse_nums(_current_numbers_str_game24(self.x, y))
        return len(nums) <= 1

    def is_win(self, y: str) -> bool:
        nums = _parse_nums(_current_numbers_str_game24(self.x, y))
        return len(nums) == 1 and abs(nums[0] - 24.0) < 1e-6

    def last_step_preview(self, y: str) -> str:
        sl = _game24_step_lines(y)
        return sl[-1] if sl else "<root>"


class _GenericAdapter:
    """
    Generic for text/crosswords:
      - depth = number of non-empty lines in y
      - key = (hash(prefix), depth) to limit memory
      - terminal = depth >= task.steps
      - win = False (no early stop), unless task defines custom hooks
    """
    def __init__(self, task):
        self.task = task

    def depth(self, y: str) -> int:
        return len(_nonempty_lines(y))

    def key(self, y: str) -> Tuple[str, int]:
        d = self.depth(y)
        # hash only first N chars to keep key stable but bounded
        prefix = y[:2000]
        h = hashlib.sha1(prefix.encode("utf-8", errors="ignore")).hexdigest()
        return h, d

    def is_terminal(self, y: str) -> bool:
        steps = getattr(self.task, "steps", 1)
        return self.depth(y) >= steps

    def is_win(self, y: str) -> bool:
        return False

    def last_step_preview(self, y: str) -> str:
        lines = _nonempty_lines(y)
        return lines[-1] if lines else "<root>"


def _pick_adapter(task, x: str):
    # If game24-like traces exist (left: ...) or task name indicates game24 -> use Game24 adapter
    name = getattr(task, "name", "") or getattr(task, "__class__", type("X", (), {})).__name__
    if "game24" in str(name).lower():
        return _Game24Adapter(x)
    if "(left:" in x:
        return _Game24Adapter(x)
    return _GenericAdapter(task)


@dataclass
class Node:
    y: str  # raw trajectory text (same style as bfs.py)
    N: int = 0
    W: float = 0.0
    children: Dict[str, "Node"] = field(default_factory=dict)  # key = child.y (raw)
    untried: Optional[List[str]] = None  # list of child trajectories (raw)

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


def solve(args, task, idx, to_print=True):
    """
    MCTS over trajectories
    Works for game24; also runs for text/crosswords with GenericAdapter
    """

    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)

    iters = getattr(args, "mcts_iters", 200)
    c = getattr(args, "mcts_c", 1.4)
    pw_k = getattr(args, "mcts_pw_k", 2.0)
    pw_alpha = getattr(args, "mcts_pw_alpha", 0.5)

    x = task.get_input(idx)
    A = _pick_adapter(task, x)

    # Transposition table (for GenericAdapter it's still OK, but less "meaningful")
    tt: Dict[object, Node] = {}

    def get_or_make_node(y: str) -> Node:
        key = A.key(y)
        n = tt.get(key)
        if n is not None:
            return n
        n = Node(y=y)
        tt[key] = n
        return n

    root = get_or_make_node("")

    def eval_leaf(y: str) -> float:
        # terminal reward for game24; for other tasks it just uses value heuristic
        if A.is_terminal(y):
            return 1.0 if A.is_win(y) else 0.0

        v = get_value(task, x, y, args.n_evaluate_sample, cache_value=True)

        # Normalize legacy scale to [0,1] for MCTS.
        try:
            vv = float(v)
        except Exception:
            vv = 0.0

        v01 = vv / 20.0 if vv >= 1.0 else vv
        if v01 < 0.0:
            return 0.0
        if v01 > 1.0:
            return 1.0
        return v01

    def ensure_untried(node: Node):
        if node.untried is not None:
            return

        props = get_proposals(task, x, node.y)

        # Dedup: avoid duplicate raw and avoid duplicate canonical keys
        seen_raw = set(node.children.keys())
        seen_key = {A.key(ch_y) for ch_y in node.children.keys()}

        out: List[str] = []
        for p in props:
            if not p or p in seen_raw:
                continue
            k = A.key(p)
            if k in seen_key:
                continue
            seen_raw.add(p)
            seen_key.add(k)
            out.append(p)

        node.untried = out
        if _dbg():
            print(f"[MCTS] ensure_untried depth={A.depth(node.y)} got={len(out)}", flush=True)

    def can_expand(node: Node) -> bool:
        ensure_untried(node)
        if not node.untried:
            return False
        limit = pw_k * ((node.N + 1) ** pw_alpha)
        return len(node.children) < limit

    def expand_one(node: Node) -> Node:
        ensure_untried(node)
        y_child = node.untried.pop(0)
        child = get_or_make_node(y_child)
        node.children[child.y] = child
        if _dbg():
            print(f"[MCTS] EXPAND d={A.depth(node.y)} -> d={A.depth(child.y)} | {A.last_step_preview(child.y)}",
                  flush=True)
        return child

    def select_child_ucb(node: Node) -> Node:
        logN = math.log(node.N + 1)
        best_score = -1e18
        best_child: Optional[Node] = None

        for ch in node.children.values():
            if ch.N == 0:
                score = 1e18
            else:
                score = ch.Q + c * math.sqrt(logN / ch.N)
            if score > best_score:
                best_score = score
                best_child = ch

        # node.children non-empty => best_child not None
        return best_child  # type: ignore

    best_found: Optional[str] = None

    for _ in range(iters):
        node = root
        path = [node]

        # Selection + Expansion with per-iteration cycle guard
        seen_in_this_iter = set()

        while True:
            key = A.key(node.y)
            if key in seen_in_this_iter:
                if _dbg():
                    print(f"[MCTS] CYCLE DETECTED key={key} depth={A.depth(node.y)} last={A.last_step_preview(node.y)}",
                          flush=True)
                break
            seen_in_this_iter.add(key)

            if A.is_terminal(node.y):
                break

            ensure_untried(node)

            if can_expand(node):
                node = expand_one(node)
                path.append(node)
                break

            if not node.children:
                break

            node = select_child_ucb(node)
            path.append(node)

        # Evaluation
        if _dbg():
            # for game24 print numbers; for others just print last step preview
            if isinstance(A, _Game24Adapter):
                nums = _current_numbers_str_game24(x, node.y)
                print(f"[MCTS] EVAL depth={A.depth(node.y)} nums=[{nums}]", flush=True)
            else:
                print(f"[MCTS] EVAL depth={A.depth(node.y)} last=[{A.last_step_preview(node.y)}]", flush=True)

        v = eval_leaf(node.y)

        if v >= 1.0 and A.is_win(node.y):
            best_found = node.y

        # Backprop
        for nd in path:
            nd.N += 1
            nd.W += v

        if best_found is not None:
            break

    def extract_greedy_path(start: Node, max_depth: int) -> str:
        """
        IMPORTANT: must not loop forever when TT merges states.
        Guards:
          - cycle detection via seen_y_keys
          - require depth progress
        """
        cur = start
        y = cur.y
        seen = {A.key(y)}

        while True:
            d = A.depth(y)
            if d >= max_depth or A.is_terminal(y):
                break

            ensure_untried(cur)

            if cur.children:
                nxt = max(cur.children.values(), key=lambda n: n.N)
            elif cur.untried:
                nxt = get_or_make_node(cur.untried[0])
                cur.children.setdefault(nxt.y, nxt)
            else:
                break

            # guard: no self-loop / no cycle
            k2 = A.key(nxt.y)
            if k2 in seen or nxt is cur:
                if _dbg():
                    print(f"[MCTS] greedy_path cycle/stall at depth={d} last={A.last_step_preview(y)}", flush=True)
                break

            # guard: must progress in depth
            d2 = A.depth(nxt.y)
            if d2 <= d:
                if _dbg():
                    print(f"[MCTS] greedy_path no depth progress d={d} -> d2={d2}", flush=True)
                break

            cur = nxt
            y = cur.y
            seen.add(k2)

        return y

    top_k = getattr(args, "n_select_sample", 5)
    steps = getattr(task, "steps", 3)

    candidates: List[str] = []
    if best_found is not None:
        candidates.append(best_found)

    ensure_untried(root)
    if not root.children and root.untried:
        for _ in range(min(top_k, len(root.untried))):
            expand_one(root)

    ranked_children = sorted(root.children.values(), key=lambda n: n.N, reverse=True)
    for ch in ranked_children[:top_k]:
        candidates.append(extract_greedy_path(ch, max_depth=steps))

    # Dedup preserve order
    seen_raw = set()
    ys: List[str] = []
    for y in candidates:
        if y not in seen_raw:
            seen_raw.add(y)
            ys.append(y)

    if not ys:
        ys = [""]

    ys = ys[:top_k]

    if to_print:
        print(gpt)
        print(f"[MCTS] idx={idx} iters={iters} root_children={len(root.children)} tt_size={len(tt)}")
        for i, y in enumerate(ys):
            nd = tt.get(A.key(y))
            if nd is None:
                print(f"#{i} N=? Q=?\n{y}")
            else:
                print(f"#{i} N={nd.N} Q={nd.Q}\n{y}")

    return ys, {"mcts": {"iters": iters, "tt_size": len(tt)}}


def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)
    return [""], {"idx": idx, "x": x}
