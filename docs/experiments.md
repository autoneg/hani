# Experiments

An **experiment** is a named, self-contained bundle of everything one study
needs: its consent text, its scenarios, its questionnaires, its scenario order
and its results. Experiments make it possible to run several studies from one
HANI installation without their configuration or data touching each other.

!!! note "Opt-in — nothing changes until you use it"
    When no experiment is selected, **every** path HANI uses is computed exactly
    as it always was (settings under `~/negmas/hani/settings/`, results under
    `~/negmas/hani/db/`). The experiment layer only kicks in once
    `HANI_EXPERIMENT` / `--experiment` is set.

## Creating an experiment

```bash
hani experiment create my-study
```

This scaffolds `~/hani/experiments/my-study/` from your current defaults:

```
~/hani/experiments/my-study/
├── consent.md                          # copied from your settings (or package default)
├── scenario_order.txt                  # copied from your settings (or package default)
├── scenarios/                          # seeded from the bundled sample scenarios
│   ├── Default/
│   └── ...
├── questionnaires/
│   └── per_negotiation.yaml            # copied if one is found; otherwise drop yours here
└── results/                            # this experiment's data is written here
```

Use `--force` (`-f`) to overwrite files that already exist. Existing files are
left alone by default, so re-running the command is safe.

## Running an experiment

Select the experiment at **startup** — it is server-wide, not per session:

```bash
# any of these are equivalent
HANI_EXPERIMENT=my-study hani
hani-app   --experiment my-study
hani-guest --experiment my-study
```

| Where | How |
|-------|-----|
| Environment | `HANI_EXPERIMENT=my-study` |
| `hani-app` | `--experiment my-study` |
| `hani-guest` | `--experiment my-study` |

!!! warning "Not on the `hani` sub-commands"
    `hani main` and `hani guest` do not (yet) take `--experiment`. Use the
    `HANI_EXPERIMENT` environment variable, or call `hani-app` / `hani-guest`
    directly.

## What becomes experiment-scoped

With an experiment active, HANI resolves each artefact through this precedence
chain (first hit wins):

1. `~/hani/experiments/<name>/<file>` — the experiment
2. `~/negmas/hani/settings/<file>` — your shared user settings
3. `<package>/default_settings/<file>` — the bundled defaults

| Artefact | Experiment-scoped location |
|----------|---------------------------|
| Consent form | `<exp>/consent.md` |
| Scenario order | `<exp>/scenario_order.txt` |
| Scenarios | `<exp>/scenarios/` — wins over `HANI_PROLIFIC_SCENARIOS_BASE`, so an experiment is fully self-contained |
| Per-negotiation questionnaire | `<exp>/questionnaires/per_negotiation.yaml` (an explicit `PROLIFIC_PER_NEG_YAML` still wins) |
| Results root | `<exp>/results/` — instead of `~/negmas/hani/db/` |

Authentication, LLM settings, Support Agent settings and Autopilot settings stay
**global** (they live in `~/negmas/hani/settings/`), since they describe the
deployment rather than the study.

## Combining with Prolific

Experiments and the [Prolific integration](prolific.md) compose: run the guest
app with both, and each participant's per-PID directory lands under the
experiment's results root.

```bash
HANI_EXPERIMENT=my-study \
PROLIFIC_N_REQUIRED=5 \
PROLIFIC_PER_NEG_YAML=~/hani/experiments/my-study/questionnaires/per_negotiation.yaml \
hani-guest --port 5008
```

Per-participant data is then written to
`~/hani/experiments/my-study/results/prolific_<PID>/` (`prolific_session.json`,
`schedule.json`, `results.csv`, `last_scenario.txt` — see
[Prolific Integration](prolific.md#files-hani-writes-per-prolific-user)).

Agent group files an external tool drops for you are also read from the active
results root: `<results>/groups/<key>.json` (see
[Agents & Opponent Pools](agents.md#participant-group-files)).

## Data collected per negotiation

| Data | Where |
|------|-------|
| One row per negotiation (utilities, agreement, `status`, `ended_by`, `practice`, `human_advantage`, timing) | `results.csv` in the participant's directory |
| Questionnaire answers | next to `results.csv` |
| Negotiation trace (what was offered) | the session's trace files |
| LLM prompts/responses of the opponent agent (why it offered) | `<user_path>/llm_logs/<mechanism_id>.jsonl`, one file per negotiation |
| Support Agent actions | event log, `component="SupportAgent"`, `value.actor="agent"` |

### Per-negotiation LLM call logging

Set `HANI_LLM_LOG=1` to record every `litellm.completion` call the opponent
agent makes during a negotiation — prompt, full response, hidden reasoning if
the model emits any, model name, latency and token budget — appended to **one
JSONL file per negotiation** beside that negotiation's other data.

```bash
HANI_LLM_LOG=1 hani-guest --port 5008
```

Where the trace tells you *what* was offered, this tells you *why*, which is
what you need to diagnose an empty or degenerate agent turn after the fact.
Logging is fully env-gated, per-session isolated (concurrent participants never
cross-write), and best-effort: a logging error can never disturb a live
negotiation.

## Event tracking & the analytics dashboard

Independently of the named experiments above, HANI records UI and negotiation
**events** through its event-tracking layer. To explore them:

```bash
hani-analysis
```

This launches the Panel-based **Event Analytics Dashboard** in your browser.

The event database has its own, older notion of an "experiment" (a labelled
window of event data), managed by a **separate** CLI:

```bash
hani-experiment create "Pilot A" --description "first pilot"
hani-experiment list --active-only
hani-experiment info <experiment_id>
hani-experiment end <experiment_id>
```

!!! warning "Two different things named 'experiment'"
    `hani experiment create <name>` (a sub-command of `hani`) creates the
    **directory-based experiment** described on this page — consent,
    scenarios, questionnaires, results.
    `hani-experiment` (a separate executable) labels windows of **event-tracking
    data** in the event database. They are unrelated; don't mix them up.
