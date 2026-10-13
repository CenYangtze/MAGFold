import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional, Protocol

from openai import OpenAI

from .utils import load_config


CONFIG = load_config("configs/configs.yaml")
LLM_CONFIG = CONFIG.get("llm_config", {})
MAX_TOKEN = LLM_CONFIG.get("max_token", 512)
TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
NUM_COMPS = LLM_CONFIG.get("num_comps", 1)
REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", LLM_CONFIG.get("request_timeout", 120)))
GARBLED_FILTER_ENV = "LLM_ENABLE_GARBLED_FILTER"
ENABLE_THINKING_ENV = "LLM_ENABLE_THINKING"

completion_tokens, prompt_tokens = 0, 0
judge_completion_tokens, judge_prompt_tokens = 0, 0


def _max_non_space_run(text: str) -> int:
    max_run = 0
    current = 0
    for char in text:
        if char.isspace():
            max_run = max(max_run, current)
            current = 0
        else:
            current += 1
    return max(max_run, current)


def _looks_like_json_response(text: str) -> bool:
    stripped = (text or "").strip()
    return (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
        or stripped.startswith("```json")
    )


def is_garbled_response(text: Optional[str]) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    is_json = _looks_like_json_response(stripped)
    if not is_json and not stripped.isascii():
        return True
    return len(stripped) >= 80 and not is_json and _max_non_space_run(stripped) >= 20


def garbled_filter_enabled() -> bool:
    return os.getenv(GARBLED_FILTER_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _requires_non_streaming_thinking_disabled(model_name: str) -> bool:
    normalized = (model_name or "").lower().replace("_", "-")
    return "qwen3" in normalized or "qwen-3" in normalized


class HardRequestTimeout:
    """Best-effort wall-clock timeout for Linux OpenAI-compatible calls."""

    def __init__(self, seconds: float):
        self.seconds = float(seconds or 0)
        self.enabled = (
            self.seconds > 0
            and hasattr(signal, "SIGALRM")
            and hasattr(signal, "getitimer")
            and hasattr(signal, "setitimer")
            and threading.current_thread() is threading.main_thread()
        )
        self._old_handler = None
        self._old_timer = None

    def __enter__(self):
        if not self.enabled:
            return self
        self._old_handler = signal.getsignal(signal.SIGALRM)
        self._old_timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if self._old_timer is not None and self._old_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, *self._old_timer)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
        return False

    def _handle_timeout(self, signum, frame):
        raise TimeoutError(f"LLM request exceeded {self.seconds:.0f}s timeout")


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMCallable(Protocol):
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS,
    ) -> str:
        pass


class LLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS,
    ) -> str:
        pass


class GPTChat(LLM):
    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        request_timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        resolved_url = api_base or os.getenv("OPENAI_API_BASE")
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_url or not resolved_key:
            raise ValueError("OPENAI_API_BASE and OPENAI_API_KEY must be set for the OpenAI backend.")
        super().__init__(model_name=model_name)
        self.request_timeout = float(request_timeout if request_timeout is not None else REQUEST_TIMEOUT)
        self.max_retries = max(1, int(max_retries if max_retries is not None else LLM_CONFIG.get("max_retries", 5)))

        configured_enable_thinking = enable_thinking
        if configured_enable_thinking is None:
            configured_enable_thinking = LLM_CONFIG.get("enable_thinking")
            env_enable_thinking = _env_bool(ENABLE_THINKING_ENV)
            if env_enable_thinking is not None:
                configured_enable_thinking = env_enable_thinking
        if configured_enable_thinking is None and _requires_non_streaming_thinking_disabled(model_name):
            configured_enable_thinking = False

        self.extra_body = {}
        if configured_enable_thinking is not None:
            self.extra_body["enable_thinking"] = bool(configured_enable_thinking)

        self.client = OpenAI(
            base_url=resolved_url,
            api_key=resolved_key,
            timeout=self.request_timeout,
            max_retries=0,
        )

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS,
    ) -> str:
        global prompt_tokens, completion_tokens

        payload = [{"role": msg.role, "content": msg.content} for msg in messages]
        wait_time = 1
        for _ in range(self.max_retries):
            try:
                request_kwargs = {
                    "model": self.model_name,
                    "messages": payload,
                    "max_tokens": int(max_tokens or MAX_TOKEN),
                    "temperature": temperature,
                    "n": int(num_comps or NUM_COMPS),
                    "stop": stop_strs,
                }
                if self.extra_body:
                    request_kwargs["extra_body"] = self.extra_body
                with HardRequestTimeout(self.request_timeout):
                    response = self.client.chat.completions.create(**request_kwargs)

                answer = response.choices[0].message.content
                if answer is None:
                    print("Error: LLM returned None")
                    continue
                if garbled_filter_enabled() and is_garbled_response(answer):
                    print("Error: LLM returned garbled text; retrying without counting this response locally.")
                    time.sleep(wait_time)
                    continue

                if response.usage is not None:
                    prompt_tokens += response.usage.prompt_tokens
                    completion_tokens += response.usage.completion_tokens
                return answer
            except Exception as exc:
                error_message = str(exc)
                if "access_denied" in error_message.lower() or "access denied" in error_message.lower():
                    print(f"Error during API call: {error_message}")
                    break
                if "rate limit" in error_message.lower() or "429" in error_message or "timeout" in error_message.lower():
                    print(f"Error during API call: {error_message}")
                    time.sleep(wait_time)
                    continue
                print(f"Error during API call: {error_message}")
                break
        return ""


def build_llm(
    backend: str = "openai",
    model_name: str = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    request_timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    **_: object,
) -> LLMCallable:
    if backend != "openai":
        raise ValueError(f"Unsupported LLM backend '{backend}'. Use 'openai' with an OpenAI-compatible endpoint.")
    return GPTChat(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        enable_thinking=enable_thinking,
        request_timeout=request_timeout,
        max_retries=max_retries,
    )


def get_price():
    return completion_tokens, prompt_tokens, completion_tokens * 60 / 1000000 + prompt_tokens * 30 / 1000000


def get_judge_price():
    return judge_completion_tokens, judge_prompt_tokens, judge_completion_tokens * 60 / 1000000 + judge_prompt_tokens * 30 / 1000000
