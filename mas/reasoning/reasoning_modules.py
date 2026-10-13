from attr import dataclass
from mas.llm import LLMCallable, Message

@dataclass
class ReasoningConfig:
    temperature: float = None
    max_tokens: int = None
    stop_strs: list[str] = None
    num_comps: int = None


def env_aware_reasoning_config(env, default: "ReasoningConfig") -> "ReasoningConfig":
    """Return a per-env reasoning config.

    Honors `env.disables_stop_newline`: when True, drops `stop_strs=['\\n']` so
    the model can produce a complete multi-line answer in one call. Useful for
    closed-context tasks (e.g. HotpotQA Distractor) where there are no
    informative intermediate observations and the stateless agent would
    otherwise regenerate the same opening Thought every iteration.
    """
    if getattr(env, "disables_stop_newline", False):
        return ReasoningConfig(
            temperature=default.temperature,
            max_tokens=default.max_tokens or 1024,
            stop_strs=None,
            num_comps=default.num_comps,
        )
    return default

class ReasoningBase:
    def __init__(self, llm_model: LLMCallable):
        self.llm_model: LLMCallable = llm_model
    
    def __call__(self, prompts: list[Message], config: ReasoningConfig) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")



class ReasoningIO(ReasoningBase):
    """
    A subclass of ReasoningBase that defines how to perform a reasoning call
    just by using an underlying language model (LLM).
    """

    def __call__(self, prompts: list[Message], config: ReasoningConfig) -> str:
        
        reasoning_result = self.llm_model(
            prompts,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop_strs=config.stop_strs,
            num_comps=config.num_comps
        )
        return reasoning_result