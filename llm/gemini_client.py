"""Old name of llm/client.py, kept because the patched Optimus-1 (official/.../gpt4_planning.py) imports it."""
from client import *  # noqa: F401,F403
from client import LLMClient as GeminiClient  # noqa: F401
