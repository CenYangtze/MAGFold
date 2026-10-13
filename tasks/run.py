import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field

import yaml
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs import BaseEnv, BaseRecorder, get_env, get_recorder, get_task
from mas.llm import LLMCallable, build_llm, get_judge_price, get_price
from mas.mas import MetaMAS
from mas.memory import MASMemoryBase
from mas.module_map import module_map
from mas.reasoning import ReasoningBase
from mas.utils import EmbeddingFunc
from mas_workflow import get_mas
from prompts import get_dataset_system_prompt, get_task_few_shots
from utils import get_model_type


SUPPORTED_TASKS = ["alfworld", "pddl", "llabench", "hotpotqa"]
SUPPORTED_MAS = ["autogen", "macnet", "dylan"]
SUPPORTED_MEMORY = ["empty", "metagpt", "chatdev", "gmemory", "magfold"]

with open("tasks/configs.yaml", encoding="utf-8") as reader:
    CONFIG: dict = yaml.safe_load(reader)

WORKING_DIR: str = None


def _read_env_file(path: str = None) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Env file not found: {path}")

    values = {}
    with open(path, "r", encoding="utf-8") as reader:
        for raw_line in reader:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key:
                values[key] = value
    return values


def _env_bool(values: dict, name: str):
    raw = values.get(name)
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} in env file must be one of true/false/1/0/yes/no/on/off.")


def _safe_run_tag(tag: str = None) -> str:
    if not tag:
        return ""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", tag).strip("_")


@dataclass
class TaskManager:
    task_name: str
    mas_type: str
    memory_type: str
    tasks: list[dict]
    env: BaseEnv
    recorder: BaseRecorder
    mas: MetaMAS
    mas_config: dict = field(default_factory=dict)
    mem_config: dict = field(default_factory=dict)


def build_task(task: str, mas_type: str, memory_type: str, max_steps: int) -> TaskManager:
    with open(CONFIG[task]["env_config_path"], encoding="utf-8") as reader:
        env_config = yaml.safe_load(reader) or {}

    return TaskManager(
        task_name=task,
        mas_type=mas_type,
        memory_type=memory_type,
        tasks=get_task(task),
        env=get_env(task, env_config, max_steps),
        recorder=get_recorder(task, working_dir=WORKING_DIR, namespace="total_task"),
        mas=get_mas(mas_type),
        mas_config=dict(CONFIG.get(mas_type, {}) or {}),
        mem_config=dict(CONFIG.get(memory_type, {}) or {}),
    )


def get_memory_root() -> str:
    configured = os.environ.get("MAS_MEMORY_DB_ROOT") or CONFIG.get("memory_folder", "./.db")
    configured = os.path.expanduser(configured)
    if not os.path.isabs(configured):
        configured = os.path.abspath(configured)
    os.makedirs(configured, exist_ok=True)
    return configured


def build_mas(
    task_manager: TaskManager,
    reasoning: str,
    mas_memory: str,
    model_name: str,
    api_base: str = None,
    api_key: str = None,
    enable_thinking: bool = None,
    segmenter_model: str = None,
    segmenter_api_base: str = None,
    segmenter_api_key: str = None,
    segmenter_enable_thinking: bool = None,
    magfold_aux_llm_with_segmenter: bool = False,
) -> None:
    embed_func = EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    reasoning_module_type, memory_module_type = module_map(reasoning, mas_memory)

    llm_model: LLMCallable = build_llm(
        backend="openai",
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        enable_thinking=enable_thinking,
    )
    reasoning_module: ReasoningBase = reasoning_module_type(llm_model=llm_model)

    if segmenter_model is not None:
        if mas_memory != "magfold":
            raise ValueError("--segmenter_model is only supported with --mas_memory magfold.")
        segmenter_llm_model = build_llm(
            backend="openai",
            model_name=segmenter_model,
            api_base=segmenter_api_base or api_base,
            api_key=segmenter_api_key or api_key,
            enable_thinking=segmenter_enable_thinking,
        )
        task_manager.mem_config["segmenter_llm_model"] = segmenter_llm_model
        if magfold_aux_llm_with_segmenter:
            task_manager.mem_config["magfold_aux_llm_model"] = segmenter_llm_model

    mas_memory_module: MASMemoryBase = memory_module_type(
        namespace=mas_memory,
        global_config=task_manager.mem_config,
        llm_model=llm_model,
        embedding_func=embed_func,
    )

    task_manager.mas.add_observer(task_manager.recorder)
    task_manager.mas.build_system(reasoning_module, mas_memory_module, task_manager.env, task_manager.mas_config)

    if hasattr(task_manager.env, "set_judge"):
        task_manager.env.set_judge(llm_model)


def _parse_task_ids(raw_task_ids: str) -> list[int]:
    task_ids = []
    for part in raw_task_ids.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            if end < start:
                raise ValueError(f"Invalid task id range: {part}")
            task_ids.extend(range(start, end + 1))
        else:
            task_ids.append(int(part))
    if not task_ids:
        raise ValueError("--task_ids did not contain any task ids.")
    seen = set()
    deduped = []
    for task_id in task_ids:
        if task_id not in seen:
            deduped.append(task_id)
            seen.add(task_id)
    return deduped


def _parse_mem_config_json(raw_mem_config: str) -> dict:
    try:
        overrides = json.loads(raw_mem_config)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --mem_config_json: {exc}") from exc
    if not isinstance(overrides, dict):
        raise ValueError("--mem_config_json must decode to a JSON object.")
    return overrides


def run_task(task_manager: TaskManager, task_ids: list[int]) -> None:
    task_manager.recorder.dataset_begin()

    for task_id, task_config in tqdm(
        zip(task_ids, task_manager.tasks),
        total=len(task_manager.tasks),
        desc="Running Tasks",
    ):
        task_manager.recorder.task_begin(task_id, task_config)

        task_main, task_description = task_manager.mas.env.set_env(task_config)
        few_shots = get_task_few_shots(
            dataset=task_manager.task_name,
            task_config=task_config,
            few_shots_num=CONFIG[task_manager.task_name].get("few_shots_num", 0),
        )
        task_config.update(task_main=task_main, task_description=task_description, few_shots=few_shots)

        task_instruction = get_dataset_system_prompt(task_manager.task_name, task_config=task_config)
        for agent in task_manager.mas.agents_team.values():
            task_manager.recorder.log(f"------------ MAS Agent: {agent.name} ------------")
            task_manager.recorder.log(agent.add_task_instruction(task_instruction))

        reward, done = task_manager.mas.schedule(task_config)
        task_manager.recorder.task_end(reward, done)

    task_manager.recorder.dataset_end()


def _select_task_slice(task_configs: TaskManager, args: argparse.Namespace) -> list[int]:
    if args.task_ids:
        selected_task_ids = _parse_task_ids(args.task_ids)
        out_of_range = [
            task_id for task_id in selected_task_ids if task_id < 0 or task_id >= len(task_configs.tasks)
        ]
        if out_of_range:
            raise ValueError(f"--task_ids contains out-of-range ids for {len(task_configs.tasks)} tasks: {out_of_range}")
        task_configs.tasks = [task_configs.tasks[task_id] for task_id in selected_task_ids]
        return selected_task_ids

    if args.start_task < 0:
        raise ValueError("--start_task must be non-negative.")
    if args.start_task >= len(task_configs.tasks):
        raise ValueError(f"--start_task {args.start_task} is out of range for {len(task_configs.tasks)} tasks.")
    if args.start_task > 0:
        task_configs.tasks = task_configs.tasks[args.start_task:]
    if args.num_tasks > 0:
        task_configs.tasks = task_configs.tasks[: args.num_tasks]
    return list(range(args.start_task, args.start_task + len(task_configs.tasks)))


def main() -> None:
    global WORKING_DIR

    random.seed(42)
    parser = argparse.ArgumentParser(description="Run MAGFold experiments with selected MAS and memory modules.")
    parser.add_argument("--task", type=str, required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--mas_type", type=str, default="autogen", choices=SUPPORTED_MAS)
    parser.add_argument("--mas_memory", type=str, default="empty", choices=SUPPORTED_MEMORY)
    parser.add_argument("--reasoning", type=str, default="io", choices=["io"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI-compatible chat model name.")
    parser.add_argument("--env_file", type=str, default=None, help="Optional .env for OPENAI_API_BASE/OPENAI_API_KEY.")
    parser.add_argument("--api_base", type=str, default=None, help="OpenAI-compatible API base. Overrides --env_file.")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI-compatible API key. Overrides --env_file.")
    parser.add_argument("--segmenter_model", type=str, default=None, help="Use a separate OpenAI-compatible MAGFold segmenter model.")
    parser.add_argument("--segmenter_env_file", type=str, default=None, help="Optional .env for the segmenter LLM.")
    parser.add_argument("--segmenter_api_base", type=str, default=None)
    parser.add_argument("--segmenter_api_key", type=str, default=None)
    parser.add_argument("--magfold_aux_llm_with_segmenter", action="store_true")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument("--successful_topk", type=int, default=None)
    parser.add_argument("--failed_topk", type=int, default=None)
    parser.add_argument("--insights_topk", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--use_projector", action="store_true", default=None)
    parser.add_argument("--hop", type=int, default=None)
    parser.add_argument("--start_task", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=-1, help="Max number of tasks to run (-1 means all tasks).")
    parser.add_argument("--task_ids", type=str, default=None, help='Comma-separated zero-based ids/ranges, e.g. "6,10,60-80".')
    parser.add_argument("--mem_config_json", type=str, default=None, help="JSON object merged into selected memory config.")

    args = parser.parse_args()
    main_env = _read_env_file(args.env_file)
    segmenter_env = _read_env_file(args.segmenter_env_file)

    max_trials = args.max_trials
    if max_trials is None:
        max_trials = CONFIG.get(args.task, {}).get("max_steps", 50)

    memory_dir_name = args.mas_memory
    safe_tag = _safe_run_tag(args.run_tag)
    if safe_tag:
        memory_dir_name = f"{memory_dir_name}_{safe_tag}"
    WORKING_DIR = os.path.join(get_memory_root(), get_model_type(args.model), args.task, args.mas_type, memory_dir_name)
    os.makedirs(WORKING_DIR, exist_ok=True)

    task_configs = build_task(args.task, args.mas_type, args.mas_memory, max_trials)
    selected_task_ids = _select_task_slice(task_configs, args)

    cli_mas_overrides = {
        "successful_topk": args.successful_topk,
        "failed_topk": args.failed_topk,
        "insights_topk": args.insights_topk,
        "threshold": args.threshold,
        "use_projector": args.use_projector,
    }
    for key, value in cli_mas_overrides.items():
        if value is not None:
            task_configs.mas_config[key] = value

    if args.mem_config_json:
        task_configs.mem_config.update(_parse_mem_config_json(args.mem_config_json))
    task_configs.mem_config["working_dir"] = WORKING_DIR
    if args.hop is not None:
        task_configs.mem_config["hop"] = args.hop

    build_mas(
        task_configs,
        reasoning=args.reasoning,
        mas_memory=args.mas_memory,
        model_name=args.model,
        api_base=args.api_base or main_env.get("OPENAI_API_BASE"),
        api_key=args.api_key or main_env.get("OPENAI_API_KEY"),
        enable_thinking=_env_bool(main_env, "LLM_ENABLE_THINKING"),
        segmenter_model=args.segmenter_model,
        segmenter_api_base=args.segmenter_api_base or segmenter_env.get("OPENAI_API_BASE"),
        segmenter_api_key=args.segmenter_api_key or segmenter_env.get("OPENAI_API_KEY"),
        segmenter_enable_thinking=_env_bool(segmenter_env, "LLM_ENABLE_THINKING"),
        magfold_aux_llm_with_segmenter=args.magfold_aux_llm_with_segmenter,
    )
    run_task(task_configs, task_ids=selected_task_ids)

    completion_tokens, prompt_tokens, _ = get_price()
    judge_completion_tokens, judge_prompt_tokens, _ = get_judge_price()
    task_configs.recorder.log(
        f"completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}, "
        f"price={completion_tokens * 15 / 1000000 + prompt_tokens * 5 / 1000000}"
    )
    if judge_completion_tokens > 0 or judge_prompt_tokens > 0:
        task_configs.recorder.log(
            f"[judge] completion_tokens:{judge_completion_tokens}, prompt_tokens:{judge_prompt_tokens}, "
            f"price={judge_completion_tokens * 15 / 1000000 + judge_prompt_tokens * 5 / 1000000}"
        )


if __name__ == "__main__":
    main()
