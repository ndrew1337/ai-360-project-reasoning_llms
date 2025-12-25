import itertools
import heapq
import numpy as np
from functools import partial
from tot.models import gpt


# ---------- helper functions (copied from bfs.py) ----------

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
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
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]


def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]


# ---------- A* solver ----------

def solve(args, task, idx, to_print=True):
    """
    A*-style Tree-of-Thoughts search.

    - State = (partial output string y, depth step)
    - g(n) = step (depth)
    - h(n) = - value(y)  (higher LM value => lower heuristic cost)
    - f(n) = g(n) + h(n)

    We keep a priority queue of states and always expand the one
    with smallest f(n). Branching is still limited by n_select_sample,
    and selection within each expansion still uses args.method_select
    ('greedy' or 'sample') for compatibility with bfs.py.
    """
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)

    x = task.get_input(idx)
    infos = []

    # Priority queue items: (f_cost, -value, step, counter, y)
    # counter breaks ties to keep heapq stable
    open_heap = []
    counter = 0

    # root state
    start_y = ''
    start_step = 0
    start_value = 0.0  # neutral heuristic at root
    start_f = start_step + (-start_value)
    heapq.heappush(open_heap, (start_f, -start_value, start_step, counter, start_y))
    counter += 1

    # track best complete solution we see
    best_y = None
    best_value = -float('inf')

    # limit number of node expansions (for cost control)
    max_expansions = getattr(args, 'max_expansions', args.n_select_sample * task.steps)
    expansions = 0

    while open_heap and expansions < max_expansions:
        f_cost, neg_val, step, _, y = heapq.heappop(open_heap)
        cur_value = -neg_val

        # if this state is already at max depth, treat as completed candidate
        if step >= task.steps:
            if cur_value > best_value:
                best_value = cur_value
                best_y = y
            continue

        # ----- generate children for this state -----
        if args.method_generate == 'sample':
            new_ys = get_samples(
                task,
                x,
                y,
                args.n_generate_sample,
                prompt_sample=args.prompt_sample,
                stop=task.stops[step],
            )
        elif args.method_generate == 'propose':
            new_ys = get_proposals(task, x, y)
        else:
            raise ValueError(f'method_generate {args.method_generate} not recognized')

        ids = list(range(len(new_ys)))

        # ----- evaluate children -----
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        else:
            raise ValueError(f'method_evaluate {args.method_evaluate} not recognized')

        # ----- choose which children actually enter the open set -----
        if len(ids) == 0:
            continue

        if args.method_select == 'sample':
            v = np.array(values, dtype=float)
            # avoid division by zero if all values are zero
            if v.sum() == 0:
                ps = np.ones_like(v) / len(v)
            else:
                ps = v / v.sum()
            k = min(args.n_select_sample, len(ids))
            select_ids = np.random.choice(ids, size=k, p=ps, replace=False).tolist()
        elif args.method_select == 'greedy':
            k = min(args.n_select_sample, len(ids))
            select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[:k]
        else:
            raise ValueError(f'method_select {args.method_select} not recognized')

        select_new_ys = [new_ys[i] for i in select_ids]
        select_values = [values[i] for i in select_ids]

        # ----- logging, similar to bfs.py -----
        if to_print:
            sorted_new_ys, sorted_values = zip(
                *sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True)
            )
            print(
                f'[step {step}] expand y="{y[:60]}..."'
                f'\n-- new_ys --: {sorted_new_ys}'
                f'\n-- sol values --: {sorted_values}'
                f'\n-- choices pushed --: {select_new_ys}\n'
            )

        infos.append({
            'step': step,
            'x': x,
            'y': y,
            'new_ys': new_ys,
            'values': values,
            'select_new_ys': select_new_ys,
        })

        # ----- push selected children into the priority queue (A* core) -----
        next_step = step + 1
        for child_y, v in zip(select_new_ys, select_values):
            g_cost = next_step
            h_cost = -float(v)  # higher value -> lower heuristic cost
            f_child = g_cost + h_cost
            heapq.heappush(
                open_heap,
                (f_child, -float(v), next_step, counter, child_y),
            )
            counter += 1

        expansions += 1

    # choose best solution: either best complete we saw,
    # or best remaining state in open set by value
    if best_y is None:
        if open_heap:
            _, neg_val, _, _, y = max(open_heap, key=lambda t: -t[1])
            best_y = y
            best_value = -neg_val
        else:
            best_y = ''
            best_value = -float('inf')

    if to_print:
        print('Best y:', best_y)
        print('Best value:', best_value)

    return [best_y], {'steps': infos}


def naive_solve(args, task, idx, to_print=True):
    """
    Keep the same naive_solve interface for compatibility.
    """
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}
