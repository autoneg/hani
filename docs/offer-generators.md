# Offer Generators

Offer generators help you create negotiation offers. They can suggest offers based on various strategies, from random sampling to AI-powered generation.

## Response Generator (LLM-based)

The Response Generator uses a large language model to create offers and text responses based on natural language instructions.

### How to Use

1. Open the **Response Generator** tool from the sidebar
2. Enter an instruction describing what you want:
   - "Propose a fair middle ground"
   - "Make a slightly better offer than my last one"
   - "Accept if the price is reasonable, otherwise counter"
3. Select the output mode:
   - **Text & Outcome** - Generate both a message and structured offer
   - **Text Only** - Generate only a message
   - **Outcome Only** - Generate only a structured offer
4. Click **Generate Response**
5. Review the generated response
6. Click **Apply to Offer** to use it

### Example Instructions

| Instruction | What it does |
|-------------|--------------|
| "Propose a fair offer" | Creates a balanced offer |
| "Improve my last offer slightly" | Makes a small concession |
| "Counter with a tough stance" | Creates an aggressive counter-offer |
| "Find a compromise on price" | Focuses on the price issue |
| "Accept if utility > 60%" | Conditional acceptance logic |

### Requirements

- LLM must be configured (see [Setting up Ollama](ollama.md))
- Negotiation must be in progress

## Utility-based Selector

The Utility-based Selector finds outcomes within a specified utility range, letting you choose offers that meet your minimum requirements.

### How to Use

1. Open the **Utility-based Selector** tool
2. Set the **Minimum Utility** slider (e.g., 70%)
3. Set the **Utility Range** slider (e.g., 10% means 70-80%)
4. Browse the table of matching outcomes
5. Click on a row to select it
6. Click **Set Offer** to apply it to the action panel

### Features

- Shows all outcomes within your specified utility range
- Displays the exact utility for each outcome
- Sortable columns for easy comparison
- Pagination for large outcome spaces

### When to Use

- You know your minimum acceptable utility
- You want to explore options in a specific range
- You need to find outcomes with particular characteristics

## Random Outcome Selector

The Random Outcome Selector generates random offers from the outcome space. Useful for exploration or when you're unsure what to offer.

### How to Use

1. Open the **Random Outcome** tool
2. Click **Set Offer** to generate and apply a random offer
3. Repeat to get different random offers

### When to Use

- Early negotiation exploration
- Breaking deadlocks with unexpected offers
- Testing different parts of the outcome space

!!! note "Admin Only"
    The Random Outcome Selector is only available to administrators.

## Comparison

| Generator | Strategy | Best For |
|-----------|----------|----------|
| Response Generator | AI-powered, context-aware | Complex negotiations, natural language |
| Utility-based Selector | Utility-constrained | Finding acceptable offers |
| Random Selector | Random sampling | Exploration, breaking deadlocks |

## Creating Custom Generators

Developers can create custom offer generators by extending the `OutcomeSelector` class:

```python
from hani.tools.tool import OutcomeSelector
from negmas import Outcome
import panel as pn


class MyCustomGenerator(OutcomeSelector):
    def get_outcome(self) -> Outcome | None:
        # Your logic to generate an outcome
        # Return a tuple of values, one per issue
        return (value1, value2, ...)

    def panel(self):
        return pn.Column(
            pn.pane.Markdown("### My Custom Generator"),
            # Your UI components
        )
```

The generator must implement:

- `get_outcome()` - Returns the generated outcome
- `panel()` - Returns the Panel UI component

See `src/hani/tools/` for complete examples.
