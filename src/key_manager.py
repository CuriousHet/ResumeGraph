"""
API Key Manager for ResumeGraph.

Manages a pool of Gemini API keys and provides automatic rotation
when a key is exhausted (HTTP 429 / ResourceExhausted).
"""
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI


class KeyManager:
    """
    Loads all GEMINI_API_KEY_* keys from environment variables and
    provides a method to invoke an LLM call with automatic key rotation
    on rate-limit errors.
    """

    def __init__(self):
        self._keys = self._load_keys()
        self._current_index = 0

        if not self._keys:
            raise RuntimeError(
                "No API keys found. Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, ... in your .env file."
            )
        print(f"[KeyManager] Loaded {len(self._keys)} API key(s).")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _load_keys(self) -> list[str]:
        """Reads all GEMINI_API_KEY_* env vars and returns them as a list."""
        keys = []
        i = 1
        while True:
            key = os.environ.get(f"GEMINI_API_KEY_{i}")
            if key is None:
                break
            keys.append(key)
            i += 1
        return keys

    def _next_key(self) -> str:
        """Returns the next key in the pool using round-robin."""
        key = self._keys[self._current_index]
        self._current_index = (self._current_index + 1) % len(self._keys)
        return key

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_llm(self, model: str, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
        """
        Creates a ChatGoogleGenerativeAI instance using the next available
        API key in the rotation pool.
        """
        api_key = self._next_key()
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
        )

    def invoke_with_retry(
        self,
        model: str,
        temperature: float,
        prompt: str,
        structured_output_schema=None,
        max_retries: int = None,
    ):
        """
        Attempts to invoke the LLM. If a rate-limit error (429) is hit,
        rotates to the next API key and retries.

        Args:
            model: Gemini model name (e.g. 'gemini-2.5-flash')
            temperature: LLM temperature
            prompt: The prompt string to send
            structured_output_schema: Optional Pydantic class for structured output
            max_retries: How many keys to try before giving up (defaults to total keys)

        Returns:
            The LLM response (raw or structured)
        """
        if max_retries is None:
            max_retries = len(self._keys)

        last_error = None

        for attempt in range(max_retries):
            api_key = self._next_key()
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key,
            )

            if structured_output_schema:
                llm = llm.with_structured_output(structured_output_schema)

            try:
                result = llm.invoke(prompt)
                return result

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(
                    keyword in error_str
                    for keyword in ["429", "resource exhausted", "rate limit", "quota"]
                )

                if is_rate_limit and attempt < max_retries - 1:
                    print(
                        f"[KeyManager] Key #{self._current_index} hit rate limit. "
                        f"Rotating to next key... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(1)  # Brief pause before retry
                    last_error = e
                    continue
                else:
                    # Not a rate limit error, or we've exhausted all keys
                    raise e

        # If we get here, all keys were exhausted
        raise RuntimeError(
            f"All {max_retries} API keys exhausted due to rate limits. Last error: {last_error}"
        )


# ------------------------------------------------------------------ #
# Module-level singleton so all nodes share the same rotation state
# ------------------------------------------------------------------ #
_manager: KeyManager | None = None


def get_key_manager() -> KeyManager:
    """Returns the singleton KeyManager instance."""
    global _manager
    if _manager is None:
        _manager = KeyManager()
    return _manager
