# Agents & Opponent Pools

HANI's negotiation partner is drawn from a **pool** of negotiator types. You can
set that pool for the whole server, for a single browser session, or pick one
specific opponent as an admin.

## Specifying negotiators

The pool is a comma-separated list of *agent specs*. Each spec is either a
negotiator name/dotted path, or a **group token** starting with `:`.

```bash
# explicit negotiators
hani --agents "AspirationNegotiator,BoulwareTBNegotiator,helpers.AgentK"

# a group
hani --agents ":template"

# groups and literals mixed
hani --agents ":llm,helpers.Atlas3,file:mynegotiator.MyNegotiator"
```

### Namespaces

| Prefix | Resolves to | Example |
|--------|-------------|---------|
| *(none)* | NegMAS negotiators (`negmas.sao`) | `AspirationNegotiator` |
| `helpers.` | HANI's bundled helper negotiators | `helpers.AgentK` |
| `genius.` | Genius agents (needs a Genius bridge) | `genius.Atlas3` |
| *(none)* | negmas-llm negotiators | `LLMHybridNegotiator` |
| `file:` | A class loaded from a Python file | `file:agents/custom/myneg.MyNegotiator` |
| dotted path | Any importable class | `negmas.sao.RandomNegotiator` |

### Group tokens

| Token | Pool |
|-------|------|
| `:llm` | All LLM-based negotiators |
| `:template` | All template-based negotiators |
| `:negmas` | All NegMAS negotiators |
| `:hani` | HANI's own negotiators |
| `:genius` | Genius negotiators |
| `:notext` | Boulware / Conceder / Linear / Hybrid **with no text component** |
| `:custom` | Whatever `HANI_CUSTOM_AGENTS` (comma-separated specs) contains |
| `:pan` | The PAN pool (personality-adjusted negotiators, `HSHP`/`HSLP`/`LSHP`/`LSLP`) — requires `pan.py` to be importable |

Group tokens expand and de-duplicate while preserving order, so
`--agents ":template,:notext"` is a valid union.

`:pan` looks for `pan.py` under `$HANI_CODE_BASE` (server default `~/hani`) and
`$SCMLWEB_CODE_BASE/python` (server default `~/scmlweb/python`), then the
current directory. When it isn't there, HANI prints a warning and stays usable —
the pool simply resolves empty.

## Per-session pools (`?agents=` / `?group=`)

The opponent pool can also be overridden **per browser session** via query
arguments, so a launcher can hand different participants different pools
without restarting the server:

    https://your-host.example/app?agents=BoulwareTBNegotiator,ConcederTBNegotiator
    https://your-host.example/app?group=:notext
    https://your-host.example/app?group=official

`?agents=` takes a literal spec list (group tokens allowed); `?group=` takes a
single group token **or** the key of a participant group file.

### Participant group files

For externally-defined pools ("the official finalists", "cohort B", …), an
external tool may drop a JSON file at `<results-root>/groups/<key>.json`, where
`<results-root>` is `~/negmas/hani/db/` — or the active experiment's `results/`
directory (see [Experiments](experiments.md)). HANI only ever **reads** it; it
never calls another service or reads a competition database.

Two shapes are accepted:

```json
["AspirationNegotiator", "helpers.AgentK", "mypkg.mymodule.MyAgent"]
```

```json
{
  "syspath": ["/abs/path/to/unzipped/agent/packages"],
  "agents": ["compA.agent.CompetitorA", "compB.agent.CompetitorB"]
}
```

The object form is what you want for unzipped competitor submissions: the listed
directories are appended to `sys.path` so the dotted specs import.

A bare `?group=<key>` first looks for a group file with that key; if there is
none, the token is treated as a literal agent spec.

## Default pools

The pool normally comes from `--agents` / `?agents=` / `?group=` / the configured
`partner_types`. When **none** of those selects anything, HANI falls back to:

| Session | Fallback pool |
|---------|---------------|
| Prolific participant | `:pan` (when `pan.py` is importable) |
| Anyone else (incl. the public playground) | Non-LLM template negotiators ("with text"), so a casual visit never incurs LLM/Ollama cost |

Prolific finalists configured via `PROLIFIC_FINALISTS` are always added to the
partner list, so a `schedule.json` entry can dispatch to them.

## Admin mode: opponent badge & picker

Admin mode is on for real admins (`is_admin`) and for any page opened with
`?admin=1` — the query flag lets an admin launcher enable the extra UI for a
guest session without an admin login. In admin mode you get:

- an **ADMIN** badge in the top bar;
- an **opponent pill** beside the title showing the current partner's
  agent name and class;
- an **opponent picker** above the Load button that overrides the per-load
  random pick. The choice is persisted per user (`admin_next_opponent.txt`), so
  it survives the page reload that Load triggers. Leave it on *auto* to
  randomize each round.

Non-admins never see partner agent types — in Prolific sessions this is
deliberate, and the admin-only sidebar cards are hidden rather than merely
disabled.

## Robustness

- **Agent LLM failures are survivable**: when an LLM-driven opponent errors or
  returns nothing, HANI diagnoses and keeps the negotiation moving instead of
  dying.
- **Local Ollama fallback**: an Ollama call that the default endpoint reports as
  "model not found" is retried once against the local daemon
  (`$HANI_OLLAMA_LOCAL_BASE`, default `http://localhost:11434`). See
  [Setting up Ollama](ollama.md#local-fallback).
- **Practice rounds** forbid the agent from ending the negotiation, so a
  participant's practice round can't be cut short.
- **Per-negotiation LLM logging** (`HANI_LLM_LOG=1`) captures the prompts and
  raw responses behind every agent turn — see
  [Experiments](experiments.md#per-negotiation-llm-call-logging).
