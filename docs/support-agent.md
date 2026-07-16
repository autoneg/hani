# Negotiation Support Agent

The **Negotiation Support Agent** is an optional, LLM-driven assistant that helps a
human negotiator during a live negotiation. It is **enabled by default**; the
participant can switch it on/off from the floating panel. In Prolific sessions it
starts **off by default** (still user-toggleable).

It appears as a small **chat-bubble icon at the bottom-right of the page**. Clicking
the bubble opens a chat panel; from there the participant can turn the agent on/off
(the bubble changes colour to reflect the state) and narrow down or re-expand its
capabilities — up to the admin-granted ceiling.

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

## Enabling the agent

Who sees the assistant is resolved from three sources, **highest precedence first**:

1. **Command line** — `--support-agent on|off|auto` on `hani`, `hani-app`, or
   `hani-guest`.
2. **Environment variable** — `HANI_SUPPORT_AGENT=on|off|auto` (what the CLI flag sets).
3. **Settings file** — the `mode` field in `support_agent_settings.json`
   (`on` / `off` / `auto`), or the legacy `enabled` bool.

The three values mean:

| Value | Who gets the assistant |
|-------|------------------------|
| `on` | Everyone. |
| `off` | Nobody. |
| `auto` | **Admins only**. |

So **out of the box the assistant is on for everyone** (except Prolific sessions,
which start with the user switch off). Examples:

```bash
hani --support-agent on      # everyone gets it
hani-app --support-agent off # force off for this deployment
HANI_SUPPORT_AGENT=on hani-guest   # enable it in the playground (guests aren't admins)
```

## Configuring the agent

Configuration lives in `~/negmas/hani/settings/support_agent_settings.json` (falls back
to the bundled default, `mode: "on"`). Admins can also edit it live from the
**“Support Agent (Admin)”** card in the sidebar (the **“Assistant for”** selector maps
to `mode`).

```json
{
  "mode": "on",
  "agent_class": null,
  "provider": "openai",
  "model": "gpt-4o-mini",
  "api_key_env": "OPENAI_API_KEY",
  "api_base": null,
  "temperature": 0.3,
  "max_tokens": 2000,
  "max_tool_iterations": 6,
  "autonomy": "full",
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
| `mode` | `on` / `off` / `auto` (admins only). Default is `on`. Overridden by `--support-agent` / `HANI_SUPPORT_AGENT`. (Legacy `enabled: true/false` is still honoured.) |
| `agent_class` | Optional dotted path to a custom agent class (see [below](#building-a-custom-support-agent)). Empty/`null` → the built-in agent. |
| `provider` / `model` | Any litellm provider and model. |
| `api_key_env` | Name of the **environment variable** holding the API key (not needed for Ollama). The key itself is never stored in the settings file. |
| `api_base` | Base URL for the host/port, e.g. `http://localhost:11434` for a local server. |
| `temperature` | Sampling temperature passed to the LLM. |
| `max_tokens` | Max output tokens per completion (mapped to the provider-specific alias). |
| `max_tool_iterations` | Safety bound on the tool-calling loop per turn (default 6). |
| `autonomy` | The **admin ceiling**: `suggest`, `semi`, or `full` (see below). Default `full`. |
| `capabilities` | The admin-granted ceiling per capability (keys = `Capability` values). |
| `proactive` | Per-event flags (`on_negotiation_started`, `on_action_requested`) enabling unprompted turns. |
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
* **User preferences** — from the **“⚙️ Capabilities”** panel inside the floating chat,
  the human negotiator can switch **off** any granted capability (and turn granted
  ones back **on**) and **lower** the autonomy level, plus a master on/off switch for
  the whole agent.

The **effective** permission is the intersection: a participant can always make the
agent *less* powerful, but never *more* powerful than the admin allowed.

Participants can also choose where **proactive** assistant messages appear:
either as toast notifications or on the always-visible bottom status board. This
setting only affects unprompted/event-triggered messages; direct chat replies
remain in chat.

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
| Floating UI | `hani.support_agent.floating_ui.build_floating_agent` | The bottom-right chat bubble + popup: a `pn.chat.ChatInterface`, the master on/off switch, and the capability/autonomy controls. Mounted into the template header. |
| Settings | `hani.support_agent.settings` | Load/save the JSON config. |
| Admin UI | `hani.support_agent.admin_ui` | The sidebar card. |

The view never owns the runtime — the agent must be able to act (toasts, tool control,
proactive comments) regardless of which tool tab is focused. All UI mutations are funnelled
through `runtime.run_on_doc(...)` / `runtime.post(...)`, which marshal onto the Bokeh
session document so nothing mutates widgets from a worker thread.

---

## Building a custom Support Agent

You can fully replace the built-in agent. **Developing a new one is subclassing
`SupportAgent` (and optionally `ToolDispatcher`) and overriding the knobs below**, then
pointing the settings at your class.

```python
# my_pkg/my_agent.py
from hani.support_agent.runtime import SupportAgent

class MyAgent(SupportAgent):
    def build_system_prompt(self) -> str:
        return "You are a terse negotiation coach. " + super().build_system_prompt()
```

```json
// support_agent_settings.json (or the admin card)
{ "enabled": true, "agent_class": "my_pkg.my_agent:MyAgent" }
```

HANI resolves the dotted path (`module:Class` or `module.Class`) via
`resolve_agent_class`, verifies it is a `SupportAgent` subclass, and instantiates it for
the session. **If it can't be loaded or isn't a `SupportAgent`, HANI logs a warning and
falls back to the built-in agent** — a bad override can never take the feature down.

!!! note "Capability and autonomy always apply"
    Custom agents still go through the same `CapabilityState`, action API, and marshalling
    layer. Your overrides change *behaviour*, never *permissions* — the admin ceiling and
    the participant's narrowing are always enforced.

---

## Override reference — every knob

This is the complete surface an overrider can touch. Everything is optional; override only
what you need and call `super()` where it makes sense.

### A. `SupportAgent` — overridable methods

| Method | Signature | Default behaviour | Override to… | Returns |
|--------|-----------|-------------------|--------------|---------|
| `build_system_prompt` | `() -> str` | Returns the `system_prompt` setting. | Change the agent's persona/instructions. | The system prompt string. |
| `make_dispatcher` | `() -> ToolDispatcher` | `self.dispatcher_class(self)`. | Build a custom dispatcher (alternative to setting `dispatcher_class`). | A `ToolDispatcher`. |
| `handle_user_message` | `(text: str) -> None` | Gated on `user_enabled`; injects the state snapshot and runs the tool-calling loop. | Take full control of a chat turn. | — (post replies via `self.post`). |
| `on_event` | `(event_name: str, nmi=None) -> None` | If `user_enabled` and `proactive["on_<event>"]` is set and not busy, runs `proactive_note`. | Change which lifecycle events trigger proactive turns / what they do. | — |
| `proactive_note` | `(event_name: str) -> str \| None` | Canned notes for `negotiation_started` / `action_requested`. | Change the seed text for proactive turns (return `None` to stay silent). | A system note, or `None`. |
| `warn_if_draft_below_reserved` | `() -> None` | Reserved-value warning rule (toast + board + chat), debounced. Runs **on the IOLoop**. | Change/disable the deterministic draft-vs-reserved check. | — |
| `_complete` | `(tools: list[dict]) -> dict` | One `litellm.completion` using the settings. | Swap the LLM backend entirely. | `{"role":"assistant","content":str,"tool_calls":[...]}`. |
| `_context_snapshot` | `() -> str \| None` | JSON snapshot of `actions["context"]()`, injected each turn. | Change what live state is fed to the model. | A string note, or `None`. |
| `_run_turn` | `(user_text=None, system_note=None, context_note=None)` | The busy-locked tool-calling loop itself. | Rarely — to re-shape the whole loop. | — |
| `_execute_tool_call` | `(tc: dict) -> None` | Runs one tool call via the dispatcher, appends the tool result. | Intercept/transform tool calls. | — |

### B. `SupportAgent` — class attribute

| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `dispatcher_class` | `type[ToolDispatcher]` | `ToolDispatcher` | Set to your `ToolDispatcher` subclass to add/replace LLM tools. |

### C. Helpers available on `self` (call these from your overrides)

| Helper | Signature | Use |
|--------|-----------|-----|
| `self.post` | `(text, user="Support Agent") -> None` | Post a chat message (unprompted-safe; never re-triggers the callback). |
| `self.toast` | `(message, level="info") -> bool` | Toast the user; `level` ∈ `info\|success\|warning\|error`. |
| `self.show_on_board` | `(text, append=False) -> bool` | Write markdown/HTML to the always-visible bottom board. |
| `self.run_on_doc` | `(fn, timeout=30.0)` | Run a UI mutation on the session document and get its result (safe from a worker thread; **don't** call from the IOLoop). |
| `self._post_async` | `(fn) -> None` | Fire-and-forget UI mutation (safe from either thread). |

### D. Instance state you can read/use

| Attribute | Type | Notes |
|-----------|------|-------|
| `self.session_state` | `dict` | The whole session (see the keys table below). |
| `self.settings` | `dict` | The merged settings dict. |
| `self.capabilities` | `CapabilityState` | Effective capability/autonomy resolver. |
| `self.dispatcher` | `ToolDispatcher` | The tool dispatcher. |
| `self.messages` | `list[dict]` | The running conversation (OpenAI message format). |
| `self.user_enabled` | `bool` | The participant's master on/off (don't widen permissions through it). |

### E. `ToolDispatcher` — add or change LLM tools

Subclass it (and set `dispatcher_class`) to expose extra functions to the model. Built-in
tools are unaffected.

| Member | Signature | Purpose |
|--------|-----------|---------|
| `register_extra_tools` | `() -> None` | **Override** this hook to register your tools (called at construction). |
| `register` | `(name, handler, schema, capability=None)` | Register one tool. `handler(args: dict) -> dict`; `schema` is a zero-arg callable returning a function schema; `capability` is a `Capability` to gate it (or `None` = always available). |
| `make_function_schema` | `(name, description, properties, required) -> dict` | Module-level helper to build an OpenAI/litellm function schema. |
| `schemas` | `() -> list[dict]` | The schemas advertised to the LLM (only *effective* capabilities). |
| `execute` | `(name, args) -> dict` | Dispatch a tool call (re-checks the capability). |

Inside handlers: `self.runtime` (the agent), `self.ss` (session_state), `self.caps`
(the `CapabilityState`). Marshal UI effects with `self.runtime.run_on_doc(...)`.

```python
from hani.support_agent.tools import ToolDispatcher, make_function_schema

class MyDispatcher(ToolDispatcher):
    def register_extra_tools(self):
        self.register(
            "summarise_history", self._summarise,
            lambda: make_function_schema("summarise_history",
                "Summarise the negotiation so far.", {}, []),
            capability=None,
        )
    def _summarise(self, args: dict) -> dict:
        return {"ok": True, "state": self.ss["actions"]["context"]()}

class MyAgent(SupportAgent):
    dispatcher_class = MyDispatcher
```

### F. Built-in LLM tools (what the model can already call)

Each is advertised only when its capability is *effective*.

| Tool | Capability gate | Effect |
|------|-----------------|--------|
| `send_toast(message, level)` | `TOAST` | Toast the user. |
| `show_on_board(content, append?)` | none (always) | Write markdown/HTML to the bottom board. |
| `get_negotiation_state()` | none (always) | Read the full live state (same as the injected snapshot). |
| `list_tools()` | any tool-control cap | List tool tabs + state. |
| `set_tool_enabled(name, enabled)` | `TOOL_ENABLE` | Enable/disable a tool. |
| `set_tool_visible(name, visible)` | `TOOL_VISIBILITY` | Show/hide a tool tab. |
| `move_tool(name, position?, pane?)` | `TOOL_ORDER` | Reorder / move a tool between panes. |
| `set_offer(outcome)` | `FILL_OFFER` | Fill the offer widgets (no submit). |
| `set_message_text(text)` | `FILL_TEXT` | Fill the message to the partner (no submit). |
| `submit_counter_offer(outcome?, message?)` | `SUBMIT_COUNTER` | Counter-offer (submit vs recommend per autonomy). |
| `accept_offer()` | `ACCEPT` | Accept (execute / confirm / recommend per autonomy). |
| `end_negotiation()` | `END` | End (execute / confirm / recommend per autonomy). |

### G. The action API — `session_state["actions"]`

The shared operations the **human buttons and the agent both call** (so behaviour can't
drift). Available once a negotiation panel exists.

| Key | Signature | Use |
|-----|-----------|-----|
| `set_offer` | `(outcome: dict) -> dict` | Fill the per-issue widgets from `{issue_name: value}`. |
| `set_text` | `(text: str) -> dict` | Fill the message field sent to the partner. |
| `get_offer` | `() -> dict` | The current draft outcome. |
| `get_partner_offer` | `() -> Outcome \| None` | The partner's current offer. |
| `can_act` | `() -> bool` | Whether it's the human's turn. |
| `context` | `() -> dict` | Full negotiation state (see snapshot keys below). |
| `highlight` | `(key) -> dict` | Highlight a button (`"accept"`/`"reject"`/`"end"`). |
| `accept` | `(confirm=True) -> …` | Accept (confirm pops the dialog; `False` executes). |
| `reject_counter` | `() -> …` | Submit the counter-offer (reads widgets + text). |
| `end` | `(confirm=True) -> …` | End the negotiation. |

### H. `context()` snapshot keys (injected each turn, and via `get_negotiation_state`)

`issues`, `partner_offer`, `partner_offer_utility`, `draft_offer`, `draft_offer_utility`,
`draft_message`, `reserved_value`, `is_my_turn`, `step`, `n_steps`, `relative_time`,
`history` (recent offers/messages, each `{by, offer, message}`).

### I. `session_state` keys the agent uses

| Key | Set by | Meaning |
|-----|--------|---------|
| `support_agent` | runtime | The live `SupportAgent` instance. |
| `support_chat` | floating UI | The `ChatInterface` (the agent posts here). |
| `support_board` | floating UI | The bottom board `Markdown` pane. |
| `support_session_id` | floating UI | Session id used for attribution logging. |
| `actions` | the app's `action_panel` | The shared action API (table G). |
| `tool_controller` | the app | The `ToolController` (tool show/hide/order). |
| `doc` | the app | The Bokeh document used for marshalling. |

### J. Capabilities & autonomy (read-only enforcement)

| Symbol | Values / methods | Notes |
|--------|------------------|-------|
| `Capability` | `CHAT, TOAST, TOOL_ENABLE, TOOL_VISIBILITY, TOOL_ORDER, FILL_OFFER, FILL_TEXT, SUBMIT_COUNTER, ACCEPT, END` | `.value` matches the settings `capabilities` keys. |
| `Autonomy` | `SUGGEST, SEMI, FULL` | Effective level = the lower of admin & user. |
| `CapabilityState.has` | `(cap) -> bool` | Effective = admin-granted **and** not user-disabled. |
| `CapabilityState.autonomy_allows_execute` | `(cap) -> bool` | Whether to execute vs only recommend. |
| `CapabilityState.autonomy_allows_confirm` | `(cap) -> bool` | Whether accept/end may pop the confirm dialog. |

Proactive event names passed to `on_event` / `proactive_note`: `"negotiation_started"`,
`"action_requested"` (gated by the matching `proactive` setting).

---

## Autopilot

**Autopilot** lets an agent make *all* of the human's negotiation decisions
(accept/reject/end, counter-offers, offers) with no human input. It is a
**separate concern** from the Support Agent's capabilities and is configured in
`autopilot_settings.json` (its own file, so it works even when the Support Agent
is disabled). A top-bar **Autopilot** switch (present in both the simple and full
views) engages it; while on, the action panel is hidden and the history keeps
updating on its own.

The autopilot **driver** — who actually decides — is selectable:

| Driver | Behaviour |
|--------|-----------|
| **Support Agent** | The LLM Negotiation Support Agent decides every turn. Autopilot temporarily forces it to **FULL** autonomy and requires it to call an action tool each turn (restored when switched off). Requires the agent to be enabled for the session. |
| a negotiator from **`AUTOPILOT_NEGOTIATORS`** | A plain negmas / negmas-llm negotiator plays the human's seat directly (no LLM Support Agent needed). Instantiated with the human's utility function. |

`AUTOPILOT_NEGOTIATORS` (in `hani.support_agent.autopilot`) defaults to a set of
pure-negmas negotiators plus the negmas-llm / template negotiators the app
exposes; override the list to change the offered drivers.

### Who chooses the driver

Resolved by mode: **admin-fixed** for authenticated / Prolific sessions (only the
switch is shown; the driver comes from the `driver` setting), and
**user-selectable** for admins and in the guest playground (a driver `Select`
appears beside the switch, limited to `allowed_drivers`).

### Settings — `autopilot_settings.json`

| Field | Meaning |
|-------|---------|
| `allowed` | Admin master switch (the "admin-allows" leg). The Autopilot switch is greyed for everyone when `false`. |
| `driver` | The admin-fixed default driver (`"Support Agent"` or a negotiator name). |
| `allowed_drivers` | Subset offered in the user-facing `Select` (`null` = offer all). |
| `step_delay` | Seconds to pause between autopilot steps (0 = as fast as possible). |
| `end_in_practice` | Reserved: whether an autopilot driver may END during practice rounds. |

The switch is enabled only under the three-way gate: **admin allows** AND a
**driver resolves** AND (the driver is a negotiator **OR** the Support Agent is
enabled this session). Admins can edit all of the above live from the
**"Autopilot (Admin)"** sidebar card.

### Stall safety (Support-Agent driver)

With no human watching, an autopilot turn that fails to act would stall the
round. The Support-Agent driver therefore forces a tool call each turn and, if
the agent still doesn't act (or the LLM errors / isn't configured), a **stall
net** sends a counter to keep the negotiation moving and, after repeated
failures, **auto-disengages** autopilot and restores the action panel so the
human is back in control.
