# Negotiation Support Agent

The **Negotiation Support Agent** is an optional, LLM-driven assistant that helps a
human negotiator during a live negotiation. It is **disabled by default** and
admin-gated; when off, it has zero effect on the app and on existing experiments.

When enabled it can:

1. **Chat** with the user in a full chat interface — and post messages *unprompted*
   (e.g. a proactive observation when the partner makes an offer).
2. **Send toast notifications** at any time.
3. **Control tools** — show/hide, enable/disable, and reorder any analysis tool.
4. **Drive the action panel** — fill the offer fields, write the message to the
   partner, and (subject to its granted *autonomy*) send a counter-offer, accept, or
   end the negotiation.

It is built as a **multi-turn tool-calling agent** on top of
[`litellm`](https://github.com/BerriAI/litellm), so it works with any provider
(OpenAI, Anthropic, Ollama, Azure, Gemini, …) and has its **own** configuration —
separate from the single-shot LLM used by the [Offer Generators](offer-generators.md).

---

## Enabling and configuring the agent

Configuration lives in `~/negmas/hani/settings/support_agent_settings.json` (falls back
to the bundled default, which has the agent **disabled**). Admins can also edit it live
from the **“Support Agent (Admin)”** card in the sidebar.

```json
{
  "enabled": true,
  "agent_class": null,
  "provider": "openai",
  "model": "gpt-4o-mini",
  "api_key_env": "OPENAI_API_KEY",
  "api_base": null,
  "temperature": 0.3,
  "max_tokens": 2000,
  "max_tool_iterations": 6,
  "autonomy": "suggest",
  "capabilities": {
    "chat": true, "toast": true,
    "tool_enable": true, "tool_visibility": true, "tool_order": true,
    "fill_offer": true, "fill_text": true,
    "submit_counter": true, "accept": true, "end": true
  },
  "proactive": {
    "on_negotiation_started": false,
    "on_action_requested": false
  },
  "system_prompt": "…"
}
```

| Field | Meaning |
|-------|---------|
| `enabled` | Master switch. When `false`, nothing is loaded. |
| `agent_class` | Optional dotted path to a custom agent class (see [below](#building-a-custom-support-agent)). Empty/`null` → the built-in agent. |
| `provider` / `model` | Any litellm provider and model. |
| `api_key_env` | Name of the **environment variable** holding the API key (not needed for Ollama). The key itself is never stored in the settings file. |
| `api_base` | Base URL for the host/port, e.g. `http://localhost:11434` for a local server. |
| `autonomy` | The **admin ceiling**: `suggest`, `semi`, or `full` (see below). |
| `capabilities` | The admin-granted ceiling per capability. |
| `proactive` | Lifecycle events on which the agent may act unprompted. |
| `system_prompt` | The agent's persona/instructions. |

### Autonomy levels

Autonomy governs how far the agent may go on **negotiation** actions:

| Level | Counter-offer | Accept / End |
|-------|---------------|--------------|
| `suggest` | Fills the offer + message and **highlights** the Send button; the human presses it. | Highlights the button only. |
| `semi` | **Auto-submits** the counter-offer. | Pops the existing confirmation dialog for the human to approve. |
| `full` | Auto-submits. | **Executes** immediately, no confirmation. |

Chat, toasts, tool control, and filling fields are always “execute” — autonomy only
restricts the irreversible negotiation moves.

### The capability lattice (admin grants, user narrows)

There are two layers:

* **Admin grant** — the *ceiling* set in settings (`capabilities` + `autonomy`).
* **User preferences** — from the **“⚙️ Agent controls”** panel inside the chat tool,
  the human negotiator can switch **off** any granted capability and **lower** the
  autonomy level.

The **effective** permission is the intersection: a participant can always make the
agent *less* powerful, but never *more* powerful than the admin allowed.

---

## Attribution / logging

Every action the agent takes is logged through the normal `event_tracking` system under
`component="SupportAgent"` with `value={"actor": "agent", …}`, so analysis can cleanly
separate agent-initiated actions from the human's own clicks.

---

## How it works (architecture)

| Piece | Module | Role |
|-------|--------|------|
| Runtime | `hani.support_agent.runtime.SupportAgent` | Owns the conversation, the LLM tool-calling loop, and the thread-safe marshalling of all UI effects. Runs the (blocking) LLM call off the IOLoop. |
| Capabilities | `hani.support_agent.capabilities` | `Capability`, `Autonomy`, `CapabilityState` (the admin∩user lattice). |
| Tools | `hani.support_agent.tools.ToolDispatcher` | The functions exposed to the LLM, capability-gated, dispatched to the action API / tool controller. |
| Tool controller | `hani.support_agent.tool_controller.ToolController` | Enable/disable, show/hide, reorder tabs (also powers the human move/close buttons). |
| Chat view | `hani.tools.support_agent_tool.SupportAgentTool` | The “Assistant” tab: a `pn.chat.ChatInterface`, the user controls panel, and lifecycle → proactive forwarding. |
| Settings | `hani.support_agent.settings` | Load/save the JSON config. |
| Admin UI | `hani.support_agent.admin_ui` | The sidebar card. |

The view never owns the runtime — the agent must be able to act (toasts, tool control,
proactive comments) even when its tab isn't focused. All UI mutations are funnelled
through `runtime.run_on_doc(...)` / `runtime.post(...)`, which marshal onto the Bokeh
session document so nothing mutates widgets from a worker thread.

---

## Building a custom Support Agent

You can fully replace the built-in agent. **Developing a new one is just subclassing
`SupportAgent` and overriding a few methods**, then pointing the settings at your class.

### 1. Subclass `SupportAgent`

```python
# my_pkg/my_agent.py
from hani.support_agent.runtime import SupportAgent


class MyAgent(SupportAgent):
    # --- override any of these (all optional) ---

    def build_system_prompt(self) -> str:
        """Customise the agent's persona/instructions."""
        return "You are a terse, no-nonsense negotiation coach. " + super().build_system_prompt()

    def proactive_note(self, event_name: str) -> str | None:
        """What (if anything) seeds an unprompted turn on a lifecycle event.
        Return None to stay silent. Requires the matching `proactive` flag in settings.
        """
        if event_name == "action_requested":
            return "(system) The partner just moved. Give one short tactical tip."
        return None

    def handle_user_message(self, text: str) -> None:
        """Full control of a chat turn. Default runs the litellm tool-calling loop."""
        return super().handle_user_message(text)
```

Helpers available on `self`:

- `self.post(text)` — post a chat message to the user (unprompted-safe).
- `self.toast(message, level)` — `level` ∈ `info|success|warning|error`.
- `self.run_on_doc(fn)` — run a UI mutation on the session document and get its result.
- `self.session_state`, `self.settings`, `self.capabilities`, `self.dispatcher`.
- `self.session_state["actions"]` — the shared action API the human buttons also use:
  `set_offer(dict)`, `set_text(str)`, `get_offer()`, `get_partner_offer()`,
  `can_act()`, `context()`, `accept(confirm=...)`, `reject_counter()`, `end(confirm=...)`.

### 2. (Optional) Add custom LLM tools

Subclass the dispatcher to expose extra functions to the model. They are advertised and
executed exactly like the built-ins (and you can capability-gate them):

```python
from hani.support_agent.tools import ToolDispatcher, make_function_schema


class MyDispatcher(ToolDispatcher):
    def register_extra_tools(self):
        self.register(
            "summarise_history",
            self._summarise_history,                       # handler(args) -> dict
            lambda: make_function_schema(
                "summarise_history",
                "Summarise the negotiation so far for the user.",
                {}, [],
            ),
            capability=None,                               # None = always available
        )

    def _summarise_history(self, args: dict) -> dict:
        ctx = self.ss["actions"]["context"]()              # read negotiation state
        return {"ok": True, "history": ctx.get("partner_offer")}


class MyAgent(SupportAgent):
    dispatcher_class = MyDispatcher
```

### 3. (Optional) Replace the LLM backend entirely

Override `_complete(self, tools)` to call any backend you like. Return a dict shaped
like an assistant message:

```python
class MyAgent(SupportAgent):
    def _complete(self, tools: list[dict]) -> dict:
        # self.messages is the running conversation (OpenAI message format).
        ...
        return {"role": "assistant", "content": "…", "tool_calls": []}
```

### 4. Point HANI at your class

Install your package in the same environment, then set in
`support_agent_settings.json` (or via the admin card):

```json
{ "enabled": true, "agent_class": "my_pkg.my_agent:MyAgent" }
```

HANI resolves the dotted path (`module:Class` or `module.Class`), verifies it is a
`SupportAgent` subclass, and uses it. **If the class can't be loaded or isn't a
`SupportAgent`, HANI logs a warning and falls back to the built-in agent** — a bad
override can never take the feature down.

!!! note "Capability and autonomy still apply"
    Custom agents go through the same `CapabilityState` and action API, so the admin
    ceiling and the participant's narrowing are always enforced — your overrides change
    *behaviour*, not *permissions*.
