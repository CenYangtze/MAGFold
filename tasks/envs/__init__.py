import json
import os

from .base_env import BaseEnv, BaseRecorder


TASKS_PATH = {
    "alfworld": "data/alfworld/alfworld_tasks_suffix.json",
    "pddl": "data/pddl/test.jsonl",
    "hotpotqa": "data/hotpotqa/hotpot_dev_distractor_v1.json",
    "llabench": "data/llabench",
}

DATASET_HINTS = {
    "alfworld": (
        "Install/download ALFWorld and place the ALFWorld task suffix file at "
        "data/alfworld/alfworld_tasks_suffix.json. The ALFWorld game/logic files "
        "should live under data/alfworld/ as documented in README.md."
    ),
    "pddl": (
        "Place the PDDL difficulty annotation file at data/pddl/test.jsonl. "
        "PDDL domains/problems are provided by the pddlgym dependency."
    ),
}


def _require_file(path: str, dataset: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {dataset} data file: {path}. {DATASET_HINTS.get(dataset, '')}")


def _load_json(path: str, dataset: str):
    _require_file(path, dataset)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_alfworld_tasks() -> list[dict]:
    from .alfworld_env import get_env_name_from_gamefile, prefixes

    path = TASKS_PATH["alfworld"]
    rows = _load_json(path, "alfworld")
    return [
        {
            "task": f'{row["goal"]}',
            "env_kwargs": {
                "config": "alfworld",
                "gamefile": row["gamefile"],
            },
            "task_type": prefixes[get_env_name_from_gamefile(row["gamefile"])],
            "env_name": get_env_name_from_gamefile(row["gamefile"]),
        }
        for row in rows
    ]


def _build_pddl_tasks() -> list[dict]:
    from .pddl_env.pddl_env import get_all_environment_configs

    task_names = ["barman", "blockworld", "gripper", "tyreworld"]
    path = TASKS_PATH["pddl"]
    _require_file(path, "pddl")
    return get_all_environment_configs(task_names, path)


def _get_hotpotqa_tasks() -> list[dict]:
    """Load HotpotQA in the official Distractor setting.

    The cached JSON mirrors HotpotQA's official `hotpot_dev_distractor_v1.json`
    layout. If the file is absent, this helper downloads the validation split
    from HuggingFace into the ignored data/ directory.
    """
    path = TASKS_PATH["hotpotqa"]

    needs_download = not os.path.exists(path)
    if not needs_download:
        try:
            preview = _load_json(path, "hotpotqa")
            if not preview or "context" not in preview[0]:
                needs_download = True
        except Exception:
            needs_download = True

    if needs_download:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading HotpotQA distractor validation data via HuggingFace ...")
        from datasets import load_dataset

        ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
        items = []
        for row in ds:
            titles = row["context"]["title"]
            sentences = row["context"]["sentences"]
            sf_titles = row["supporting_facts"]["title"]
            sf_ids = row["supporting_facts"]["sent_id"]
            items.append(
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "type": row["type"],
                    "level": row["level"],
                    "context": [[t, list(s)] for t, s in zip(titles, sentences)],
                    "supporting_facts": [[t, i] for t, i in zip(sf_titles, sf_ids)],
                }
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    data = _load_json(path, "hotpotqa")
    return [
        {
            "task": item["question"],
            "answer": item["answer"],
            "env_name": "hotpotqa",
            "context": item.get("context", []),
            "type": item.get("type", ""),
            "level": item.get("level", ""),
        }
        for item in data
    ]


def _build_llabench_tasks() -> list[dict]:
    from .lifelongagentbench_env import build_lifelongagentbench_tasks

    return build_lifelongagentbench_tasks()


def get_env(task: str, env_config: dict, max_trials: int) -> BaseEnv:
    if task == "alfworld":
        from .alfworld_env import AlfworldEnv

        return AlfworldEnv(env_config, max_trials)
    if task == "pddl":
        from .pddl_env.pddl_env import PDDLEnv

        return PDDLEnv(env_config, max_trials)
    if task == "hotpotqa":
        from .hotpotqa_env import HotpotQAEnv

        return HotpotQAEnv(env_config, max_trials)
    if task == "llabench":
        from .lifelongagentbench_env import LifelongAgentBenchEnv

        return LifelongAgentBenchEnv(env_config, max_trials)
    raise ValueError(f"Unsupported task type: {task}")


def get_recorder(task: str, working_dir: str, namespace: str) -> BaseRecorder:
    if task == "alfworld":
        from .alfworld_env import AlfworldRecorder

        return AlfworldRecorder(working_dir=working_dir, namespace=namespace)
    if task == "pddl":
        from .pddl_env.pddl_env import PDDLRecorder

        return PDDLRecorder(working_dir=working_dir, namespace=namespace)
    if task == "hotpotqa":
        from .hotpotqa_env import HotpotQARecorder

        return HotpotQARecorder(working_dir=working_dir, namespace=namespace)
    if task == "llabench":
        from .lifelongagentbench_env import LifelongAgentBenchRecorder

        return LifelongAgentBenchRecorder(working_dir=working_dir, namespace=namespace)
    raise ValueError(f"Unsupported task type: {task}")


def get_task(task: str) -> list[dict]:
    if task == "alfworld":
        return _build_alfworld_tasks()
    if task == "pddl":
        return _build_pddl_tasks()
    if task == "hotpotqa":
        return _get_hotpotqa_tasks()
    if task == "llabench":
        return _build_llabench_tasks()
    raise ValueError(f"Unsupported task type: {task}")
