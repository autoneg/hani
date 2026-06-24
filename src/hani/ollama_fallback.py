"""Local-ollama fallback for LLM calls.

On the deployment server the default ollama endpoint litellm talks to is the
hosted cloud (``ollama.com``), which only serves a curated set of models. An
agent (or HANI's own extraction) that asks for a model the cloud does not have
gets ``APIConnectionError: model '<name>' not found`` and the negotiation dies.
The same model is usually present in the *local* ollama daemon.

This installs a thin wrapper around ``litellm.completion`` so that an ollama
call which the default endpoint reports as "model not found" is retried once
against the local ollama daemon. It is a no-op when the first call succeeds
(e.g. on dev boxes whose default endpoint already is local), so it is safe to
always install.

negmas-llm performs the call as ``litellm.completion(**kwargs)`` (a module
attribute lookup at call time), so patching the module attribute is enough to
cover both the agents' message layer and HANI's extraction.
"""

from __future__ import annotations

import os


# litellm prefixes ollama model strings with one of these (e.g.
# "ollama_chat/qwen3:4b-instruct"). Used to scope the fallback to ollama only.
_OLLAMA_PREFIXES = ("ollama/", "ollama_chat/")


def _looks_like_model_missing(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "not found" in msg or ("model" in msg and "does not exist" in msg)


def install_ollama_local_fallback(local_base: str | None = None) -> None:
    """Patch ``litellm.completion`` to retry ollama 'model not found' locally.

    Args:
        local_base: Base URL of the local ollama daemon. Defaults to
            ``$HANI_OLLAMA_LOCAL_BASE`` or ``http://localhost:11434``.
    """
    try:
        import litellm
    except Exception:
        return
    if getattr(litellm, "_hani_local_fallback_installed", False):
        return

    base = local_base or os.getenv("HANI_OLLAMA_LOCAL_BASE", "http://localhost:11434")
    original = litellm.completion

    def completion_with_local_fallback(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            model = str(kwargs.get("model", ""))
            is_ollama = model.startswith(_OLLAMA_PREFIXES)
            api_base = str(kwargs.get("api_base") or "")
            already_local = ("localhost" in api_base) or ("127.0.0.1" in api_base)
            if is_ollama and not already_local and _looks_like_model_missing(exc):
                retry_kwargs = dict(kwargs)
                retry_kwargs["api_base"] = base
                print(
                    f"[hani] ollama model {model!r} unavailable on the default "
                    f"endpoint ({exc}); retrying against local ollama {base}"
                )
                return original(*args, **retry_kwargs)
            raise

    litellm.completion = completion_with_local_fallback
    litellm._hani_local_fallback_installed = True
