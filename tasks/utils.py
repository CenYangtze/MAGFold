def get_model_type(model_name: str) -> str:
    normalized = (model_name or "").lower()
    known_model_types = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1",
        "gpt-4",
        "gpt-3.5",
        "qwen3",
        "qwen2.5",
        "deepseek",
    ]
    for model_type in known_model_types:
        if model_type in normalized:
            return model_type.replace(".", "")
    return "unknown"
