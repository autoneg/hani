# Prolific Integration

HANI has built-in support for running paid studies on
[Prolific](https://www.prolific.com). When a participant lands on the
**guest** app (`hani-guest`) with a `PROLIFIC_PID` query parameter, HANI
switches into Prolific mode: it identifies the participant by their PID,
locks in a scenario type, enforces a per-session negotiation quota, can
follow a per-participant schedule, and ends with a completion link.

HANI provides the in-app behavior only. Study creation on Prolific,
consent/pre/post-session questionnaires, and bonus payments are the job
of whatever external service you build around HANI; that service writes
`schedule.json` (optional) and hosts the completion page that
`SCMLWEB_BASE_URL` points at.

## Entry point

The Prolific link must target the guest app:

    https://your-host.example/hanplay?PROLIFIC_PID={{%PROLIFIC_PID%}}

Run the guest app with `hani-guest` (sets `HANI_GUEST_MODE=true`):

```bash
hani-guest --port 5008
# or use the bundled wrapper for the nginx-behind setup:
./runguest.sh
```

In guest mode, HANI hides the "User Results" pane and disables the
login/consent flow — the PID is the identity.

## Session flow

A participant's session goes:

1. **First visit only** — HANI creates `~/negmas/hani/db/prolific_<PID>/`
   and writes `prolific_session.json` with `started_at`,
   `scenario_type`, `max_minutes`, `max_negs`.
2. **Practice round** — the first negotiation of a participant's first
   session uses a random partner, is flagged `practice=True` in
   `results.csv`, and does not count.
3. **Counted rounds** — the next `PROLIFIC_N_REQUIRED` (default 5)
   negotiations count. Each round pulls a (partner, scenario) cell from
   `schedule.json` if present, otherwise falls back to a random partner
   and a hash-based scenario.
4. **Zero-action rounds don't count** — if the participant lets the
   agent time out without sending anything, accepting, or ending, the
   counted-slot counter does not advance and the same opponent is shown
   again next round.
5. **Returning participants** (same PID, later session) skip the
   practice and start at counted slot 0.
6. **Completion** — when all required counted rounds are done, HANI
   shows a panel with a single link to
   `{SCMLWEB_BASE_URL}/prolific/done?PROLIFIC_PID=<pid>`.

If the participant doesn't press **Start** / **Load** within
`PROLIFIC_AUTO_START_SECONDS` (default 120), HANI auto-starts the next
round so a participant can't stall the session.

## Environment variables

All Prolific behavior is controlled by env vars. Set them in whatever
launches `hani-guest`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROLIFIC_N_REQUIRED` | `5` | Number of **counted** negotiations required. Practice is on top, so a first session has `N + 1` total rounds. |
| `PROLIFIC_MAX_MINUTES` | `45` | Recommended duration shown to the participant. **Not enforced.** |
| `PROLIFIC_AUTO_START_SECONDS` | `120` | Idle timeout before HANI auto-starts the next round. |
| `PROLIFIC_FINALISTS` | *(empty)* | Comma-separated dotted Python class names of the agents available for counted rounds. When non-empty, `schedule.json` picks from this list. Empty falls back to the configured `partner_types`. |
| `PROLIFIC_PER_NEG_YAML` | *(unset)* | Absolute path to a per-negotiation questionnaire YAML. If unset or the file is missing, the per-round form is silently skipped. |
| `SCMLWEB_BASE_URL` | `https://anac.cs.brown.edu` | Base URL for the completion link. Override with e.g. `http://localhost:8000` for local dev. |

## Per-participant schedule (`schedule.json`)

If present at `~/negmas/hani/db/prolific_<PID>/schedule.json`, HANI uses
it to dispatch counted rounds. The file is a JSON list (or an object
with a top-level `negotiations` list) of entries:

```json
{
  "negotiations": [
    {"slot": 0, "agent_class_name": "negmas.sao.AspirationNegotiator",
     "scenario_type": "Trade",  "scenario_index": 0},
    {"slot": 1, "agent_class_name": "negmas.sao.RandomNegotiator",
     "scenario_type": "Island", "scenario_index": 3}
  ]
}
```

Each entry can carry `slot` (int), `agent_class_name` (dotted Python
class, resolved against `partner_types`; falls back to random if not
found), `scenario_type` (`"Trade"` / `"Island"` / `"Grocery"`), and
`scenario_index` (int). Missing fields fall back to HANI's defaults.

When `schedule.json` is absent or has no `scenario_type`, HANI picks the
type deterministically from the PID hash:

    scenario_type = SCENARIO_LIST[ sha1(PID) mod len(SCENARIO_LIST) ]

so revisits stay on the same domain.

## Per-negotiation questionnaire

If `PROLIFIC_PER_NEG_YAML` points at a readable YAML file, HANI renders
a short form after every counted round and writes the answers next to
`results.csv`. Missing file → form is skipped. The YAML format:

```yaml
version: "2026-05-19"
title: "After this negotiation"
intro: "Quick check before we continue."

questions:
  - id: satisfaction_with_outcome
    text: "How satisfied are you with the outcome of this negotiation?"
    type: likert5             # likert5 | likert7 | yes_no | select | text
    required: true
    labels:
      1: "Not at all"
      5: "Extremely"
  - id: partner_cooperativeness
    text: "How cooperative did your partner feel?"
    type: select
    required: true
    options: ["Very competitive", "Neutral", "Very cooperative"]
```

Question text is rendered as Markdown above each widget. `likert` and
`yes_no` widgets default to **blank**, so required-field validation
fires instead of silently submitting "1".

## Files HANI writes per Prolific user

Inside `~/negmas/hani/db/prolific_<PID>/`:

- `prolific_session.json` — written on first visit.
- `schedule.json` — *(optional, written externally)* per-PID schedule.
- `results.csv` — one row per negotiation. Columns include `practice`,
  `status` (`success` / `broken` / …), `ended_by` (`human` / `agent` /
  empty), human/agent utilities, agreement, and per-round timing.
- `last_scenario.txt` — index of the last scenario shown.

## Counting rules

- **Accept** (agreement) counts.
- **End** by the participant counts.
- **Zero-action** round (agent timed out, no human action) does **not**
  count and the same opponent is shown again.

The end-of-round toast tells the participant which case applied
("timed out", "was ended by you", "was ended by the AI agent", or
"reached an agreement").

## Local test

```bash
HANI_GUEST_PORT=12000 \
PROLIFIC_N_REQUIRED=2 \
PROLIFIC_AUTO_START_SECONDS=600 \
SCMLWEB_BASE_URL=http://localhost:8000 \
./runguest.sh --port 12000

open "http://localhost:12000/hanplay?PROLIFIC_PID=test123"
```

You should see one practice round followed by two counted rounds, then
the completion link. Delete
`~/negmas/hani/db/prolific_test123/` between runs to start fresh.
