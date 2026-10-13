from typing import Any, Literal, Optional
import re
from dataclasses import dataclass

from .base_env import BaseEnv, BaseRecorder
from .utils import normalize_answer


class HotpotQAEnv(BaseEnv):
    """HotpotQA Distractor setting with closed-set Search.

    At task start the env shows only the 10 paragraph TITLES (2 supporting + 8 distractors).
    The agent reads a paragraph's content by issuing `Search[title]` and only sees the body
    as an observation. This forces real multi-hop reasoning without internet access while
    keeping the closed-context guarantee of the Distractor setting."""

    def __init__(
        self,
        env_config: dict[str, Any],
        max_trials: int = 7
    ) -> None:

        self.env_config = env_config
        self.max_trials: int = max_trials
        # Paragraph index lives across reset() because the MAS workflow calls
        # env.reset() AFTER env.set_env() — wiping the dicts here would erase the
        # task's title/content map before step() ever runs.
        self._title_to_text: dict[str, str] = {}
        self._titles: list[str] = []

        self.reset()

    def set_env(self, configs: dict) -> tuple[str, str]:
        if configs.get('answer') is None:
            raise ValueError('Please provide the answer for the question.')
        if configs.get('task') is None:
            raise ValueError('The configs dict should have the `task` attribute.')
        self.config = configs

        question: str = self.config.get('task')
        context = self.config.get('context') or []

        # Index titles for fast Search lookup later.
        self._title_to_text: dict[str, str] = {
            title: ' '.join(sentences) for title, sentences in context
        }
        self._titles: list[str] = list(self._title_to_text.keys())

        titles_block: str = '\n'.join(f'- {t}' for t in self._titles)

        task_main: str = f'Question: {question}'
        task_description: str = (
            f'Question: {question}\n\n'
            f'Available paragraphs ({len(self._titles)} titles; use Search[title] to read each):\n'
            f'{titles_block}'
        )
        return task_main, task_description

    def reset(self) -> None:
        # Reset per-trial counters only. Title/text index is owned by set_env
        # and must survive the reset() call that the MAS workflow performs
        # right after set_env().
        self.current_task: str = None
        self.reward: float = 0
        self.thought_count: int = 0
        self.recent_thoughts: list[str] = []
        # RL hook (used by AnswerJudge in rl/answer_judge.py); harmless otherwise.
        self.last_answer: Optional[str] = None

    def step(self, action: str) -> tuple[str, float, bool]:

        action: str = self.process_action(action)

        if self._parse_action_type(action) == 'thought':
            self.thought_count += 1
            # Detect content-level repetition of a Thought (paraphrasing of the same plan).
            content_only = re.sub(r'^thought\s+\d+\s*:\s*', '', action.lower())
            normalized = re.sub(r'\s+', ' ', content_only).strip()
            duplicate = normalized and normalized in self.recent_thoughts
            self.recent_thoughts.append(normalized)
            if len(self.recent_thoughts) > 5:
                self.recent_thoughts.pop(0)

            if duplicate:
                observation = (
                    "OK, but that Thought repeats a previous one. Issue Search[<title>] "
                    "to read a new paragraph, or Finish[<answer>] if you have enough evidence."
                )
            else:
                observation = 'OK.'
            return observation, 0, False

        action_type, argument = self._parse_action(action)

        if action_type and action_type.lower() == 'finish':
            # RL hook: stash the candidate answer text so the AnswerJudge can
            # grade it on a 0-10 scale. Inert for the production pipeline.
            self.last_answer = argument
            if self.success_fn(argument):
                observation = 'Answer is CORRECT'
                self.reward = 1
                return observation, 1, True
            else:
                observation = 'Answer is INCORRECT'
                return observation, 0, True

        if action_type and action_type.lower() == 'search':
            content = self._lookup_paragraph(argument)
            if content is None:
                titles_hint = ', '.join(f'"{t}"' for t in self._titles)
                observation = (
                    f'No paragraph titled "{argument}" in this task. '
                    f'Available titles are: {titles_hint}.'
                )
                return observation, -1, False
            matched_title, body = content
            return f'[{matched_title}]: {body}', 0, False

        observation = 'Invalid Action. Valid actions: Search[<title>] or Finish[<answer>]. Reasoning must use the `Thought N: ` prefix.'
        return observation, -1, False

    def _lookup_paragraph(self, query: str) -> Optional[tuple[str, str]]:
        """Resolve a Search query against the 10 titles.

        Strategy: exact case-insensitive match first, then substring match either way.
        Returns (matched_title, body) or None if no title matches.
        """
        if not query:
            return None
        q = query.strip().strip('"').strip("'").lower()
        for title in self._titles:
            if title.lower() == q:
                return title, self._title_to_text[title]
        for title in self._titles:
            if q in title.lower() or title.lower() in q:
                return title, self._title_to_text[title]
        return None

    @staticmethod
    def _parse_action_type(action: str) -> Literal['action', 'thought']:
        # Allow an optional "Action N: " prefix in front of Search/Finish.
        stripped = re.sub(r'^Action\s+\d+\s*:\s*', '', action.strip(), flags=re.IGNORECASE)
        if re.match(r'^(Search|Finish)\[', stripped, re.IGNORECASE):
            return 'action'
        return 'thought'

    @staticmethod
    def process_action(action: str) -> str:
        # Strip <think>...</think> blocks (thinking-mode models such as Qwen 3).
        if '</think>' in action:
            action = action.split('</think>', 1)[1]
        action = action.replace('<think>', '').replace('</think>', '')

        # One step per turn (workflow uses stop_strs=['\n']).
        action = action.strip().split('\n')[0].strip()
        # Some small local models emit the action as a Markdown quote, e.g.
        # ">Action 2: Search[Ed Wood]". Treat the leading quote marker as noise.
        action = re.sub(r'^>+\s*', '', action).strip()
        action = action.replace('OK.', '').replace('OK', '').strip()

        # KEEP any "Thought N: " / "Action N: " prefix as-is. The recorded trajectory
        # then carries step numbers, so on subsequent turns the stateless agent sees
        # "Thought 1: ...", "Action 1: Search[..]", "Thought 2: ..." and continues
        # counting instead of regenerating from Thought 1 every turn.
        # `_parse_action_type` / `_parse_action` below tolerate the prefix.
        return action

    @staticmethod
    def _parse_action(string: str) -> tuple[str, str]:
        # Tolerate an optional "Action N: " prefix.
        string = re.sub(r'^Action\s+\d+\s*:\s*', '', string.strip(), flags=re.IGNORECASE)
        pattern = r'^(\w+)\[(.+)\]$'
        match = re.match(pattern, string)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def success_fn(self, agent_ans: str) -> bool:
        gold = self.config.get('answer')
        return normalize_answer(agent_ans) == normalize_answer(gold)

    def feedback(self) -> tuple[float, bool, str]:

        feedback: str = 'You successfully finished this task.' if self.reward == 1 else 'You failed the task.'
        done = self.reward == 1

        return self.reward, done, feedback


@dataclass
class HotpotQARecorder(BaseRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = 'hotpotqa'
        self.counts = 0
        self.dones = 0
        self.rewards = 0

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)

        message: str = f'---------- Task: {task_id} ----------'
        self.log(message)

    def task_end(self, reward: float, done: bool):

        self.rewards += reward
        self.dones += done
        self.counts += 1

        message = f'reward: {reward}, ave reward: {self.rewards / self.counts}.\ndone: {done}, ave done: {self.dones / self.counts}'
        self.log(message)
