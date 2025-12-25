import re
import os
import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import * 

def normalize_traj(y: str) -> list[str]:
    # drop empty/whitespace-only lines; strip only the right side (keeps indentation if any)
    return [ln.rstrip() for ln in y.splitlines() if ln.strip()]


def get_current_numbers(y: str) -> str:
    lines = normalize_traj(y)
    last_line = lines[-1] if lines else ""
    return last_line.split('left: ')[-1].split(')')[0]


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    # def test_output(self, idx: int, output: str):
    #     expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
    #     numbers = re.findall(r'\d+', expression)
    #     problem_numbers = re.findall(r'\d+', self.data[idx])
    #     if sorted(numbers) != sorted(problem_numbers):
    #         return {'r': 0}
    #     try:
    #         # print(sympy.simplify(expression))
    #         return {'r': int(sympy.simplify(expression) == 24)}
    #     except Exception as e:
    #         # print(e)
    #         return {'r': 0}

    def test_output(self, idx: int, output: str):
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        if not lines:
            return {'r': 0}

        last = lines[-1].lower()

        if 'left:' in last:
            left = last.split('left:')[-1].split(')')[0].strip()
            if left == '24':
                return {'r': 1}

        # original behavior (final expression line)
        expression = last.replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception:
            return {'r': 0}
            
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
        else:
            prompt = propose_prompt.format(input=current_numbers)
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        lines = normalize_traj(y)
        last_line = lines[-1] if lines else ""

        if 'left:' not in last_line:  # last step (final expression)
            ans = last_line.lower().replace('answer: ', '')
            return value_last_step_prompt.format(input=x, answer=ans)

        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        lines = normalize_traj(y)

        # count only actual intermediate steps (the ones that contain "left:")
        step_lines = [ln for ln in lines if "left:" in ln]

        # if we've already done 4 steps and still no final "Answer:" line, don't evaluate further
        if len(step_lines) >= 4 and 'answer' not in y.lower():
            return 0

        value_names = [normalize_traj(_)[-1].strip().lower() for _ in value_outputs if normalize_traj(_)]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        return sum(score * value_names.count(name) for name, score in value_map.items())