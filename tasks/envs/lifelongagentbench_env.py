"""LLABench/LifelongAgentBench adapters for the MAGFold task/memory pipeline.

The benchmark ships three sub-environments:
  - db_bench: single-table SQL tasks
  - os_interaction: shell tasks
  - knowledge_graph: Freebase-style API tasks

This module exposes them through one BaseEnv so the existing MAS workflows and
all supported memory baselines, including MAGFold, continue to use the same lifecycle:
set_env -> retrieve memory -> step loop -> feedback -> save task context.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .base_env import BaseEnv, BaseRecorder


LAB_DATASET_NAME = "csyq/LifelongAgentBench"
LAB_PARQUET_FILES = {
    "db_bench": "db_bench/train-00000-of-00001.parquet",
    "os_interaction": "os_interaction/train-00000-of-00001.parquet",
    "knowledge_graph": "knowledge_graph/train-00000-of-00001.parquet",
}
LAB_SUB_ENV_ALIASES = {
    "db": "db_bench",
    "db_bench": "db_bench",
    "database": "db_bench",
    "os": "os_interaction",
    "os_interaction": "os_interaction",
    "kg": "knowledge_graph",
    "knowledge_graph": "knowledge_graph",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_literal(value: Any, fallback: Any = None) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return fallback if fallback is not None else value


def _normalise_lab_sub_env(name: Optional[str]) -> str:
    key = (name or "db_bench").strip().lower().replace("-", "_")
    if key not in LAB_SUB_ENV_ALIASES:
        raise ValueError(
            f"Unsupported LifelongAgentBench sub-environment: {name}. "
            f"Allowed: {sorted(set(LAB_SUB_ENV_ALIASES.values()))}"
        )
    return LAB_SUB_ENV_ALIASES[key]


def _load_lab_env_config() -> dict[str, Any]:
    path = _repo_root() / "tasks" / "env_configs" / "llabench_config.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _normalise_lab_row(sub_env: str, row: dict[str, Any]) -> dict[str, Any]:
    sample_index = int(row.get("sample_index", row.get("row_idx", 0)))
    skill_list = _safe_literal(row.get("skill_list"), []) or []

    if sub_env == "db_bench":
        answer_info = _safe_literal(row.get("answer_info"), {}) or {}
        table_info = _safe_literal(row.get("table_info"), {}) or {}
        return {
            "task": row["instruction"],
            "instruction": row["instruction"],
            "sample_index": sample_index,
            "answer_info": answer_info,
            "table_info": table_info,
            "skill_list": list(skill_list),
            "env_name": "llabench_db",
            "lab_env": sub_env,
        }

    if sub_env == "os_interaction":
        return {
            "task": row["instruction"],
            "instruction": row["instruction"],
            "sample_index": sample_index,
            "initialization_command_item": _safe_literal(row.get("initialization_command_item"), {}) or {},
            "evaluation_info": _safe_literal(row.get("evaluation_info"), {}) or {},
            "skill_list": list(skill_list),
            "raw_entry_hash": row.get("raw_entry_hash"),
            "env_name": "llabench_os",
            "lab_env": sub_env,
        }

    if sub_env == "knowledge_graph":
        return {
            "task": row["question"],
            "question": row["question"],
            "sample_index": sample_index,
            "qid": row.get("qid", ""),
            "source": row.get("source", ""),
            "entity_dict": _safe_literal(row.get("entity_dict"), {}) or {},
            "s_expression": row.get("s_expression", ""),
            "action_list": _safe_literal(row.get("action_list"), []) or [],
            "answer_list": _safe_literal(row.get("answer_list"), []) or [],
            "skill_list": list(skill_list),
            "env_name": "llabench_kg",
            "lab_env": sub_env,
        }

    raise ValueError(f"Unsupported LifelongAgentBench sub-environment: {sub_env}")


def _lab_cache_path(sub_env: str, split: str, cache_dir: str) -> Path:
    return _repo_root() / cache_dir / f"{sub_env}_{split}.json"


def _hf_parquet_repo_path(sub_env: str, split: str) -> Optional[str]:
    if split != "train":
        return None
    return LAB_PARQUET_FILES.get(sub_env)


def _hf_parquet_url(dataset_name: str, sub_env: str, split: str) -> Optional[str]:
    parquet_path = _hf_parquet_repo_path(sub_env, split)
    if parquet_path is None:
        return None
    return f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{parquet_path}"


def _env_value(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def _load_lab_parquet_fallback(load_dataset, dataset_name: str, selected: str, split: str):
    parquet_path = _hf_parquet_repo_path(selected, split)
    if parquet_path is None:
        raise RuntimeError(f"No direct parquet fallback is configured for {selected}/{split}.")

    try:
        from huggingface_hub import hf_hub_download

        local_parquet_path = hf_hub_download(
            repo_id=dataset_name,
            filename=parquet_path,
            repo_type="dataset",
        )
        return load_dataset("parquet", data_files=local_parquet_path, split="train")
    except Exception as hub_exc:
        parquet_url = _hf_parquet_url(dataset_name, selected, split)
        if parquet_url is None:
            raise
        try:
            return load_dataset("parquet", data_files=parquet_url, split="train")
        except Exception as url_exc:
            raise RuntimeError(
                "Both HuggingFace Hub download and direct parquet URL loading failed. "
                f"Hub download error: {hub_exc}. Direct URL error: {url_exc}."
            ) from url_exc


def _load_lab_dataset(load_dataset, dataset_name: str, selected: str, split: str):
    try:
        return load_dataset(dataset_name, data_dir=selected, split=split)
    except Exception as exc:
        try:
            return _load_lab_parquet_fallback(load_dataset, dataset_name, selected, split)
        except Exception as parquet_exc:
            raise RuntimeError(
                "Failed to load LifelongAgentBench from HuggingFace. The primary "
                f"data_dir load failed with: {exc}. The parquet fallback "
                f"also failed with: {parquet_exc}."
            ) from parquet_exc


def build_lifelongagentbench_tasks(sub_env: Optional[str] = None) -> list[dict]:
    """Load and cache LifelongAgentBench rows for one sub-environment.

    The HuggingFace dataset stores the three benchmark parts as separate
    parquet folders under one repo. `load_dataset(..., data_dir=sub_env)` is the
    reliable way to load them; the default config currently fails because the
    three folders have different schemas.
    """

    cfg = _load_lab_env_config()
    selected = _normalise_lab_sub_env(
        sub_env
        or _env_value("LLABENCH_ENV", "LIFELONG_AGENT_BENCH_ENV")
        or cfg.get("default_sub_env", "db_bench")
    )
    split = _env_value("LLABENCH_SPLIT", "LIFELONG_AGENT_BENCH_SPLIT", default=cfg.get("split", "train"))
    dataset_name = _env_value(
        "LLABENCH_DATASET",
        "LIFELONG_AGENT_BENCH_DATASET",
        default=cfg.get("dataset_name", LAB_DATASET_NAME),
    )
    cache_dir = _env_value(
        "LLABENCH_CACHE_DIR",
        "LIFELONG_AGENT_BENCH_CACHE_DIR",
        default=cfg.get("cache_dir", "data/llabench"),
    )
    refresh = (_env_value("LLABENCH_REFRESH", "LIFELONG_AGENT_BENCH_REFRESH", default="") or "").lower() in {
        "1",
        "true",
        "yes",
    }
    cache_path = _lab_cache_path(selected, split, cache_dir)

    if cache_path.exists() and not refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading LLABench/LifelongAgentBench from HuggingFace requires the `datasets` "
            "package. Install requirements or place a cached JSON file at "
            f"{cache_path}."
        ) from exc

    try:
        ds = _load_lab_dataset(load_dataset, dataset_name, selected, split)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load LLABench/LifelongAgentBench from HuggingFace. If you are "
            f"offline, provide a cached JSON file at {cache_path}. Detail: {exc}"
        ) from exc
    tasks = [_normalise_lab_row(selected, dict(row)) for row in ds]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    return tasks


class LifelongAgentBenchEnv(BaseEnv):
    def __init__(self, env_config: dict[str, Any], max_trials: int = 6) -> None:
        self.env_config = env_config or {}
        self.max_trials = max_trials
        self.lab_env = _normalise_lab_sub_env(
            _env_value("LLABENCH_ENV", "LIFELONG_AGENT_BENCH_ENV")
            or self.env_config.get("default_sub_env", "db_bench")
        )

        self.config: dict[str, Any] = {}
        self.reward = 0.0
        self.last_answer: Optional[str] = None

        self._db_conn: Optional[sqlite3.Connection] = None
        self._db_table_info: Optional[dict[str, Any]] = None
        self._db_answer_info: Optional[dict[str, Any]] = None

        self._os_root: Optional[Path] = None
        self._os_timeout = int(self.env_config.get("os_command_timeout", 10))

        self._kg_action_list: list[str] = []
        self._kg_answer_list: list[str] = []
        self._kg_progress = 0
        self._kg_variables: list[str] = []

    @property
    def disables_stop_newline(self) -> bool:
        return True

    def set_env(self, task_config: dict) -> tuple[str, str]:
        self.config = task_config
        self.lab_env = _normalise_lab_sub_env(task_config.get("lab_env", self.lab_env))
        self.reward = 0.0
        self.last_answer = None

        if self.lab_env == "db_bench":
            return self._set_db_env(task_config)
        if self.lab_env == "os_interaction":
            return self._set_os_env(task_config)
        if self.lab_env == "knowledge_graph":
            return self._set_kg_env(task_config)
        raise ValueError(f"Unsupported LifelongAgentBench sub-environment: {self.lab_env}")

    def reset(self) -> None:
        self.reward = 0.0
        self.last_answer = None

    @staticmethod
    def process_action(action: str) -> str:
        if "</think>" in action:
            action = action.split("</think>", 1)[1]
        action = action.replace("<think>", "").replace("</think>", "").strip()
        action = re.sub(r"^>+\s*", "", action).strip()
        action = re.sub(r"^Action\s+\d+\s*:\s*", "", action, flags=re.IGNORECASE).strip()

        act_match = re.search(r"(?is)\bAct\s*:\s*bash\b", action)
        if act_match:
            fences = re.findall(r"```(?:bash|sh)?\s*(.*?)```", action, flags=re.IGNORECASE | re.DOTALL)
            if fences:
                return f"Bash[{chr(10).join(fences).strip()}]"
        if re.search(r"(?is)\bAct\s*:\s*finish\b", action):
            return "Finish[done]"

        bracket_match = LifelongAgentBenchEnv._extract_first_bracket_action(
            action, ("Query", "SQL", "Bash", "Finish", "Answer")
        )
        if bracket_match:
            name, content = bracket_match
            if name.lower() in {"query", "sql", "finish", "answer"}:
                content = re.sub(r"(?is)^```(?:sql)?\s*(.*?)\s*```$", r"\1", content).strip()
            if name.lower() == "sql":
                name = "Query"
            return f"{name}[{content}]"

        final_answer = re.search(r"(?is)Final\s+Answer\s*:\s*(.+)\s*$", action)
        if final_answer:
            return f"Final Answer: {final_answer.group(1).strip()}"

        kg_action = re.search(r"(?is)Action\s*:\s*(\w+\s*\(.+\))", action)
        if kg_action:
            return f"Action: {kg_action.group(1).strip()}"

        return action

    def step(self, action: str) -> tuple[str, float, bool]:
        action = self.process_action(action)
        if self.lab_env == "db_bench":
            return self._step_db(action)
        if self.lab_env == "os_interaction":
            return self._step_os(action)
        if self.lab_env == "knowledge_graph":
            return self._step_kg(action)
        return "Unsupported LifelongAgentBench sub-environment.", -1, False

    def feedback(self) -> tuple[float, bool, str]:
        done = self.reward == 1
        feedback = "You successfully finished this task." if done else "You failed the task."
        return self.reward, done, feedback

    # ------------------------------------------------------------------ DB --

    def _set_db_env(self, task_config: dict) -> tuple[str, str]:
        table_info = task_config.get("table_info")
        answer_info = task_config.get("answer_info")
        if not isinstance(table_info, dict) or not isinstance(answer_info, dict):
            raise ValueError("DB LifelongAgentBench task requires table_info and answer_info.")
        self._close_db()
        self._db_table_info = table_info
        self._db_answer_info = answer_info
        self._db_conn = self._new_db_connection(table_info)

        table_name = table_info["name"]
        columns = table_info.get("column_info_list", [])
        rows = table_info.get("row_list", [])
        column_lines = "\n".join(f"- {col['name']} {col['type']}" for col in columns)
        preview_rows = rows[: min(5, len(rows))]
        preview = "\n".join(f"- {row}" for row in preview_rows)
        skills = ", ".join(task_config.get("skill_list", [])) or "unknown"

        task_main = f"[DB] {task_config['instruction']}"
        task_description = (
            "LifelongAgentBench / DB task\n"
            f"Sample index: {task_config.get('sample_index')}\n"
            f"Skills: {skills}\n\n"
            f"Instruction: {task_config['instruction']}\n\n"
            f"Table: {table_name}\n"
            f"Columns:\n{column_lines}\n"
            f"Initial row count: {len(rows)}\n"
            f"Initial row preview:\n{preview}\n\n"
            "Actions:\n"
            "- Query[<SQL>] executes SQL against the current task database and returns rows or a mutation summary.\n"
            "- Finish[<SQL>] submits the final SQL on a fresh copy of the task database for grading.\n"
            "- For mutation tasks, you may also perform Query[UPDATE/INSERT/DELETE ...] and then Finish[done]."
        )
        return task_main, task_description

    @staticmethod
    def _quote_ident(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    @classmethod
    def _new_db_connection(cls, table_info: dict[str, Any]) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        columns = table_info.get("column_info_list", [])
        table_name = cls._quote_ident(table_info["name"])
        col_defs = ", ".join(
            f"{cls._quote_ident(col['name'])} {cls._sqlite_type(col.get('type', 'TEXT'))}"
            for col in columns
        )
        conn.execute(f"CREATE TABLE {table_name} ({col_defs})")
        col_names = ", ".join(cls._quote_ident(col["name"]) for col in columns)
        placeholders = ", ".join("?" for _ in columns)
        rows = table_info.get("row_list", [])
        if rows:
            conn.executemany(
                f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})",
                [tuple(row) for row in rows],
            )
        conn.commit()
        return conn

    @staticmethod
    def _sqlite_type(type_name: str) -> str:
        upper = (type_name or "").upper()
        if "INT" in upper:
            return "INTEGER"
        if any(token in upper for token in ("REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC")):
            return "REAL"
        return "TEXT"

    def _close_db(self) -> None:
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None

    @staticmethod
    def _looks_like_sql(text: str) -> bool:
        return bool(re.match(r"(?is)^\s*(select|with|insert|update|delete|replace|create|drop|alter|pragma)\b", text))

    @staticmethod
    def _parse_bracket_action(action: str, names: tuple[str, ...]) -> tuple[Optional[str], Optional[str]]:
        match = LifelongAgentBenchEnv._extract_first_bracket_action(action.strip(), names)
        if match is None:
            return None, None
        name, content = match
        return name.lower(), content.strip()

    @staticmethod
    def _extract_first_bracket_action(action: str, names: tuple[str, ...]) -> Optional[tuple[str, str]]:
        """Return the first supported Action[...] block and ignore later lines.

        Local models sometimes emit several intended actions in one response,
        e.g. Query[...] followed by Finish[done]. The MAS can execute only one
        environment action per turn, and the sanitized action is what gets
        recorded into logs and memory, so trailing unexecuted actions must not
        survive parsing.
        """
        pattern = r"(?is)\b(" + "|".join(re.escape(n) for n in names) + r")\s*\[(.*?)\]"
        match = re.search(pattern, action)
        if not match:
            return None
        return match.group(1), match.group(2).strip()

    @classmethod
    def _run_sql(cls, conn: sqlite3.Connection, sql: str) -> tuple[bool, str, list[tuple[Any, ...]]]:
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            return False, "Empty SQL.", []
        try:
            if re.match(r"(?is)^\s*(select|with|pragma)\b", sql):
                cur = conn.execute(sql)
                rows = cur.fetchall()
                return True, cls._format_rows(rows), rows
            conn.executescript(sql + ";")
            conn.commit()
            return True, "SQL executed.", []
        except Exception as exc:
            return False, f"SQL error: {exc}", []

    @staticmethod
    def _format_rows(rows: list[tuple[Any, ...]], limit: int = 20) -> str:
        if not rows:
            return "[]"
        shown = rows[:limit]
        suffix = f"\n... ({len(rows) - limit} more rows)" if len(rows) > limit else ""
        return "\n".join(str(tuple(row)) for row in shown) + suffix

    def _step_db(self, action: str) -> tuple[str, float, bool]:
        kind, content = self._parse_bracket_action(action, ("Query", "Finish", "Answer"))
        if kind == "query":
            assert self._db_conn is not None
            ok, message, _ = self._run_sql(self._db_conn, content or "")
            return message, 0 if ok else -1, False

        if kind in {"finish", "answer"}:
            self.last_answer = content or ""
            correct = self._grade_db_finish(content or "")
            self.reward = 1.0 if correct else 0.0
            return "Answer is CORRECT" if correct else "Answer is INCORRECT", self.reward, True

        return (
            "Invalid Action. Use Query[<SQL>] to inspect or mutate the table, "
            "then Finish[<SQL>] (or Finish[done] after a mutation).",
            -1,
            False,
        )

    def _grade_db_finish(self, content: str) -> bool:
        assert self._db_table_info is not None
        assert self._db_answer_info is not None
        content = content.strip()

        if self._looks_like_sql(content):
            conn = self._new_db_connection(self._db_table_info)
            try:
                return self._grade_db_sql_on_connection(conn, content)
            finally:
                conn.close()

        if self._db_answer_info.get("direct") is not None:
            return self._validate_direct_answer_string(content, self._db_answer_info["direct"])

        if self._db_answer_info.get("md5") is not None and content.lower() in {"done", "finish", "finished", "ok"}:
            assert self._db_conn is not None
            return self._table_md5(self._db_conn, self._db_table_info) == self._db_answer_info["md5"]

        return False

    def _grade_db_sql_on_connection(self, conn: sqlite3.Connection, sql: str) -> bool:
        assert self._db_answer_info is not None
        assert self._db_table_info is not None
        first = sql.strip().split(None, 1)[0].lower() if sql.strip() else ""
        ok, _, rows = self._run_sql(conn, sql)
        if not ok:
            return False
        if first in {"select", "with", "pragma"}:
            expected = self._db_answer_info.get("direct")
            return expected is not None and self._rows_equal(rows, expected)
        expected_md5 = self._db_answer_info.get("md5")
        return expected_md5 is not None and self._table_md5(conn, self._db_table_info) == expected_md5

    @staticmethod
    def _rows_equal(actual: list[tuple[Any, ...]], expected: list[list[Any]]) -> bool:
        if len(actual) != len(expected):
            return False
        for act_row, exp_row in zip(actual, expected):
            if len(act_row) != len(exp_row):
                return False
            for act, exp in zip(act_row, exp_row):
                if isinstance(exp, (int, float)) and isinstance(act, (int, float)):
                    if not math.isclose(float(act), float(exp), rel_tol=1e-6, abs_tol=1e-6):
                        return False
                elif str(act) != str(exp):
                    return False
        return True

    @classmethod
    def _validate_direct_answer_string(cls, answer: str, expected: list[list[Any]]) -> bool:
        tuple_matches = re.findall(r"\((.*?)\)", answer, flags=re.DOTALL)
        if not tuple_matches and len(expected) == 1:
            tuple_matches = [answer]
        if len(tuple_matches) != len(expected):
            return False
        parsed_rows: list[tuple[Any, ...]] = []
        for row_text in tuple_matches:
            try:
                parsed = ast.literal_eval("(" + row_text + ")")
                if not isinstance(parsed, tuple):
                    parsed = (parsed,)
            except Exception:
                parsed = tuple(part.strip().strip("'\"") for part in row_text.split(","))
            parsed_rows.append(parsed)
        return cls._rows_equal(parsed_rows, expected)

    @classmethod
    def _table_md5(cls, conn: sqlite3.Connection, table_info: dict[str, Any]) -> str:
        columns = [col["name"] for col in table_info.get("column_info_list", [])]
        col_sql = ", ".join(cls._quote_ident(col) for col in columns)
        table_sql = cls._quote_ident(table_info["name"])
        rows = conn.execute(f"SELECT {col_sql} FROM {table_sql}").fetchall()
        row_hashes: list[str] = []
        for row in rows:
            # MySQL CONCAT_WS(',', ...) skips NULL values.
            parts = [cls._mysql_cell_to_str(value) for value in row if value is not None]
            row_hashes.append(hashlib.md5(",".join(parts).encode("utf-8")).hexdigest()[:5])
        return hashlib.md5(",".join(sorted(row_hashes)).encode("utf-8")).hexdigest()

    @staticmethod
    def _mysql_cell_to_str(value: Any) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    # ------------------------------------------------------------------- OS --

    def _set_os_env(self, task_config: dict) -> tuple[str, str]:
        self._cleanup_os_root()
        self._os_root = Path(tempfile.mkdtemp(prefix="lab_os_"))
        init = task_config.get("initialization_command_item", {})
        init_script = init.get("script", "") if isinstance(init, dict) else ""
        if init_script:
            result = self._run_os_script(init_script)
            if result.returncode != 0:
                raise RuntimeError(f"OS task initialization failed: {result.stdout}{result.stderr}")

        skills = ", ".join(task_config.get("skill_list", [])) or "unknown"
        task_main = f"[OS] {task_config['instruction']}"
        task_description = (
            "LifelongAgentBench / OS task\n"
            f"Sample index: {task_config.get('sample_index')}\n"
            f"Skills: {skills}\n\n"
            f"Instruction: {task_config['instruction']}\n\n"
            "Actions:\n"
            "- Bash[<script>] executes a shell script in an isolated task workspace.\n"
            "- Finish[done] asks the environment to run the hidden evaluation command.\n"
            "Use relative paths when possible. Benchmark absolute paths such as /var/log/app "
            "are mapped into the task workspace by this adapter."
        )
        return task_main, task_description

    def _step_os(self, action: str) -> tuple[str, float, bool]:
        kind, content = self._parse_bracket_action(action, ("Bash", "Finish"))
        if kind == "bash":
            result = self._run_os_script(content or "")
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode != 0:
                output = output or f"Command exited with code {result.returncode}."
            return output.strip() or "The output of the OS is empty.", 0, False

        if kind == "finish":
            correct = self._grade_os()
            self.reward = 1.0 if correct else 0.0
            return "Answer is CORRECT" if correct else "Answer is INCORRECT", self.reward, True

        return "Invalid Action. Use Bash[<script>] or Finish[done].", -1, False

    def _grade_os(self) -> bool:
        eval_info = self.config.get("evaluation_info", {})
        command_item = eval_info.get("evaluation_command_item", {}) if isinstance(eval_info, dict) else {}
        script = command_item.get("script", "") if isinstance(command_item, dict) else ""
        if not script:
            return False
        result = self._run_os_script(script)
        return result.returncode == 0

    def _run_os_script(self, script: str) -> subprocess.CompletedProcess[str]:
        if self._os_root is None:
            self._os_root = Path(tempfile.mkdtemp(prefix="lab_os_"))
        bash = shutil.which("bash")
        if bash is None:
            return subprocess.CompletedProcess(
                args=["bash", "-lc", script],
                returncode=127,
                stdout="",
                stderr="bash is not available on this system.",
            )

        root = str(self._os_root)
        rewritten = self._rewrite_os_script(script)
        prelude = (
            "set -e\n"
            f"export LAB_ROOT={self._shell_quote(root)}\n"
            "mkdir -p \"$LAB_ROOT/var\" \"$LAB_ROOT/home/user\" \"$LAB_ROOT/tmp\" \"$LAB_ROOT/etc\" \"$LAB_ROOT/opt\" \"$LAB_ROOT/srv\"\n"
            "cd \"$LAB_ROOT\"\n"
        )
        try:
            return subprocess.run(
                [bash, "-lc", prelude + rewritten],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._os_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", "ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", "ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return subprocess.CompletedProcess(
                args=[bash, "-lc", rewritten],
                returncode=124,
                stdout=stdout,
                stderr=stderr + f"\nCommand timed out after {self._os_timeout} seconds.",
            )

    @staticmethod
    def _shell_quote(value: str) -> str:
        return "'" + value.replace("'", "'\\''") + "'"

    @staticmethod
    def _rewrite_os_script(script: str) -> str:
        # Keep executables such as /bin/bash and /usr/bin/env untouched; map the
        # benchmark's mutable data paths into LAB_ROOT.
        for root in ("var", "home", "tmp", "etc", "opt", "srv"):
            script = re.sub(rf"(?<![\w$])/{root}(?=/|\b)", f'"$LAB_ROOT"/{root}', script)
        return script

    def _cleanup_os_root(self) -> None:
        if self._os_root is not None and self._os_root.exists():
            shutil.rmtree(self._os_root, ignore_errors=True)
        self._os_root = None

    # ------------------------------------------------------------------- KG --

    def _set_kg_env(self, task_config: dict) -> tuple[str, str]:
        self._kg_action_list = list(task_config.get("action_list", []))
        self._kg_answer_list = [str(item) for item in task_config.get("answer_list", [])]
        self._kg_progress = 0
        self._kg_variables = []

        entity_lines = "\n".join(
            f"- {name}: {mid}" for name, mid in task_config.get("entity_dict", {}).items()
        )
        api_lines = "\n".join(
            f"- {api}"
            for api in (
                "get_relations(entity_or_variable)",
                "get_neighbors(entity_or_variable, relation)",
                "intersection(var_a, var_b)",
                "get_attributes(entity_or_variable)",
                "argmax(var, attribute)",
                "argmin(var, attribute)",
                "count(var)",
            )
        )
        skills = ", ".join(task_config.get("skill_list", [])) or "unknown"

        task_main = f"[KG] {task_config['question']}"
        task_description = (
            "LifelongAgentBench / Knowledge Graph task\n"
            f"Sample index: {task_config.get('sample_index')}\n"
            f"Skills: {skills}\n\n"
            f"Question: {task_config['question']}\n"
            f"Entities:\n{entity_lines}\n\n"
            f"Available APIs:\n{api_lines}\n\n"
            "Actions:\n"
            "- Action: api_name(arguments) executes one KG API step in the lightweight trace adapter.\n"
            "- Finish[<entity_id>, ...] submits final Freebase IDs directly.\n"
            "This adapter validates final IDs locally. For exact SPARQL-backed KG exploration, "
            "wire the upstream LifelongAgentBench KG service and keep this task interface unchanged."
        )
        return task_main, task_description

    def _step_kg(self, action: str) -> tuple[str, float, bool]:
        kind, content = self._parse_bracket_action(action, ("Finish",))
        if kind == "finish":
            predicted = self._parse_kg_answer_ids(content or "")
            correct = set(predicted) == set(self._kg_answer_list)
            self.reward = 1.0 if correct else 0.0
            return "Answer is CORRECT" if correct else "Answer is INCORRECT", self.reward, True

        final_var = re.match(r"(?is)^Final Answer:\s*(?:#|var(?:iable)?\s*#?)(\d+)\s*$", action.strip())
        if final_var:
            # In the lightweight adapter, completing the hidden gold action
            # trace means the last variable denotes the answer set.
            correct = self._kg_progress >= len(self._kg_action_list)
            self.reward = 1.0 if correct else 0.0
            return "Answer is CORRECT" if correct else "Answer is INCORRECT", self.reward, True

        api_match = re.match(r"(?is)^Action:\s*(.+)$", action.strip())
        if api_match:
            api_call = re.sub(r"\s+", "", api_match.group(1))
            expected = ""
            if self._kg_progress < len(self._kg_action_list):
                expected = re.sub(r"\s+", "", self._kg_action_list[self._kg_progress])
            if expected and api_call == expected:
                var_id = len(self._kg_variables)
                self._kg_variables.append(f"#{var_id}")
                self._kg_progress += 1
                return (
                    f"Observation: API call accepted. Stored result as #{var_id}. "
                    f"Trace progress {self._kg_progress}/{len(self._kg_action_list)}.",
                    0,
                    False,
                )
            return (
                "Observation: the lightweight KG adapter could not validate that API call. "
                "Try another KG API step, or submit known Freebase IDs with Finish[...].",
                -1,
                False,
            )

        return "Invalid Action. Use Action: api_name(arguments) or Finish[<entity_id>, ...].", -1, False

    @staticmethod
    def _parse_kg_answer_ids(answer: str) -> list[str]:
        cleaned = answer.strip().strip("[]")
        if not cleaned:
            return []
        parts = re.split(r"[\s,;]+", cleaned)
        return [part.strip().strip("'\"") for part in parts if part.strip()]

    def __del__(self) -> None:
        try:
            self._close_db()
            self._cleanup_os_root()
        except Exception:
            pass


@dataclass
class LifelongAgentBenchRecorder(BaseRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = "llabench"
        self.counts = 0
        self.dones = 0
        self.rewards = 0.0

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        lab_env = task_config.get("lab_env", "unknown")
        sample_index = task_config.get("sample_index", task_id)
        self.log(f"---------- Task: {task_id} ({lab_env}:{sample_index}) ----------")

    def task_end(self, reward: float, done: bool):
        self.rewards += reward
        self.dones += int(done)
        self.counts += 1
        self.log(
            f"reward: {reward}, ave reward: {self.rewards / self.counts:.4f}.\n"
            f"done: {done}, ave done: {self.dones / self.counts:.4f}"
        )
