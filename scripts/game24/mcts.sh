python run.py \
  --task game24 \
  --task_start_index 900 \
  --task_end_index 950 \
  --search mcts \
  --method_generate propose \
  --method_evaluate value \
  --method_select greedy \
  --n_evaluate_sample 3 \
  --n_select_sample 5 \
  --mcts_iters 200 \
  --mcts_c 1.4 \
  --temperature 0.7 \
  ${@}
