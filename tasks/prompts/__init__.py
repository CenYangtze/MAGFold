from .alfworld_prompt import alfworld_solver_system_prompt, alfworld_few_shots
from .hotpotqa_prompt import hotpotqa_solver_system_prompt, hotpotqa_few_shots
from .lifelongagentbench_prompt import (
    lifelongagentbench_solver_system_prompt,
    get_lifelongagentbench_few_shots,
)
from .pddl_prompt import pddl_prompts


def get_dataset_system_prompt(task: str, task_config: dict) -> str:
    prompt_map = {
        "alfworld": alfworld_solver_system_prompt,
        "hotpotqa": hotpotqa_solver_system_prompt,
        "llabench": lifelongagentbench_solver_system_prompt,
        "pddl": pddl_prompts,
    }
    if task not in prompt_map:
        raise ValueError(f"Unsupported task type: {task}")

    if task == "pddl":
        task_type = task_config.get("game_name")
        if task_type is None:
            raise ValueError("The PDDL task config must have the `game_name` attribute.")
        return pddl_prompts[task_type]["instruction"]
    return prompt_map[task]


def get_task_few_shots(dataset: str, task_config: dict, few_shots_num: int) -> list[str]:
    if dataset == "alfworld":
        task_type = task_config.get("task_type")
        if task_type is None:
            raise ValueError("The ALFWorld task config must have the `task_type` attribute.")
        return [alfworld_few_shots[f"react_{task_type}_2"], alfworld_few_shots[f"react_{task_type}_0"]][:few_shots_num]

    if dataset == "hotpotqa":
        return hotpotqa_few_shots[:few_shots_num]

    if dataset == "llabench":
        return get_lifelongagentbench_few_shots(task_config, few_shots_num)

    if dataset == "pddl":
        task_type = task_config.get("game_name")
        if task_type is None:
            raise ValueError("The PDDL task config must have the `game_name` attribute.")
        return pddl_prompts[task_type]["examples"][:few_shots_num]

    raise ValueError(f"Unsupported dataset type: {dataset}")
