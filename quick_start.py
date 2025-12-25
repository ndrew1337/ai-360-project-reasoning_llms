import argparse
import time
from tot.methods.bfs import solve
from tot.methods.bfs import naive_solve
from tot.tasks.game24 import Game24Task

args = argparse.Namespace(
    backend='gpt-4o',      # or gpt-4.1-mini if your key supports it
    temperature=0.7,
    task='game24',
    naive_run=False,
    prompt_sample='cot',        # only used if method_generate='sample'
    method_generate='sample',   # less brittle than 'propose' for 4o-mini
    method_evaluate='value',
    method_select='greedy',
    n_generate_sample=1,
    n_evaluate_sample=3,        # from 3 -> 1
    n_select_sample=5,          # from 5 -> 2
)

task = Game24Task()
print("Calling solve()...")
t0 = time.time()
# ys, infos = solve(args, task, 900)
ys, infos = naive_solve(args, task, 900)
t1 = time.time()
print("solve() finished in", t1 - t0, "seconds")
print("Answer:", ys[0])