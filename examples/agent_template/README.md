# HANI support-agent template

A minimal, working starting point for your own negotiation support agent.

## Quick start

```bash
pip install hani                       # Python 3.13+
hani setup
ollama pull qwen3:4b-instruct          # or configure any litellm provider
cp settings.example.json ~/negmas/hani/settings/support_agent_settings.json
PYTHONPATH=$PWD hani guest             # open http://localhost:5008/app, click Start
```

Click the chat bubble (bottom right) and talk to your agent.
If it answers like the default agent, check the server log for
`could not load custom agent_class`: the folder must be on `PYTHONPATH`.

## Files

| File | Purpose |
|------|---------|
| `my_agent.py` | The agent. Edit the three `EDIT` markers. |
| `settings.example.json` | Settings that load your agent with safe, suggest-only autonomy. |
| `test_my_agent.py` | Offline smoke test (`pytest`). Extend it as your agent grows. |

Docs: <https://autoneg.github.io/hani/tutorial/> and
<https://autoneg.github.io/hani/support-agent/>.
