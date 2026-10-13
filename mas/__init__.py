import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

if os.getenv("OPENAI_API_BASE") is not None:
    os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
if os.getenv("OPENAI_API_KEY") is not None:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
