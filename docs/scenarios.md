# Defining Scenarios

Scenarios define the negotiation environment, including the issues being negotiated and the preferences of each party.

## Scenario Structure

A scenario consists of:

1. **Outcome Space** - Defines the issues and possible values
2. **Utility Functions** - Defines preferences for each party
3. **Metadata** - Optional descriptions and hints

## File Organization

Scenarios are stored in `~/negmas/hani/settings/scenarios/`:

```
scenarios/
└── CategoryName/
    └── ScenarioName/
        ├── ScenarioName.yml    # Outcome space definition
        ├── human.yml           # Human player utility function
        ├── agent.yml           # AI agent utility function
        └── _info.yaml          # Optional metadata
```

## Outcome Space Definition

The outcome space file defines the issues being negotiated.

### Example: Trade.yml

```yaml
name: Trade
type: DiscreteCartesianOutcomeSpace
issues:
  - name: Quantity
    type: ContiguousIssue
    values:
      - 1
      - 10
  - name: Price
    type: DiscreteCardinalIssue
    values:
      - 100
      - 110
      - 120
      - 200
```

### Issue Types

| Type | Description | Values Format |
|------|-------------|---------------|
| `ContiguousIssue` | Integer range | `[min, max]` |
| `DiscreteCardinalIssue` | Discrete numeric values | List of values |
| `CategoricalIssue` | Categorical options | List of strings |

### ContiguousIssue Example

```yaml
- name: Quantity
  type: ContiguousIssue
  values:
    - 1    # minimum
    - 100  # maximum
```

Creates integer values from 1 to 100.

### DiscreteCardinalIssue Example

```yaml
- name: Price
  type: DiscreteCardinalIssue
  values:
    - 50
    - 75
    - 100
    - 150
    - 200
```

Only these specific values are valid.

### CategoricalIssue Example

```yaml
- name: Color
  type: CategoricalIssue
  values:
    - Red
    - Green
    - Blue
```

## Utility Function Definition

Utility functions define how much each party values different outcomes.

### Example: Buyer.yml

```yaml
name: buyer
type: LinearAdditiveUtilityFunction
reserved_value: 0.0
weights:
  - 0.25
  - 0.75
values:
  - type: TableFun
    mapping:
      1: 0.0
      2: 0.0
      3: 0.5
      4: 1.0
      5: 0.8
      6: 0.6
  - type: TableFun
    mapping:
      100: 1.0
      110: 0.8
      120: 0.4
      200: 0.0
```

### Components

| Field | Description |
|-------|-------------|
| `name` | Identifier for the utility function |
| `type` | Usually `LinearAdditiveUtilityFunction` |
| `reserved_value` | Utility of disagreement (walking away) |
| `weights` | Importance of each issue (should sum to 1) |
| `values` | Value function for each issue |

### Value Function Types

#### TableFun

Maps specific values to utilities:

```yaml
type: TableFun
mapping:
  100: 1.0   # value 100 gives utility 1.0
  150: 0.5   # value 150 gives utility 0.5
  200: 0.0   # value 200 gives utility 0.0
```

#### LinearFun

Linear interpolation between min and max:

```yaml
type: LinearFun
min_value: 0.0    # utility at minimum
max_value: 1.0    # utility at maximum
```

#### IdentityFun

Uses the value directly (normalized):

```yaml
type: IdentityFun
```

## Metadata (_info.yaml)

Optional metadata provides descriptions and hints.

### Example

```yaml
Name: Trade
title: Simple Trading Scenario
short_description: A buyer and seller negotiate quantity and price.
long_description: |
  **The Domain**
  
  In this scenario, a buyer and seller negotiate over the quantity
  and price of goods to be traded.
  
  **Buyer's Goal**
  
  The buyer wants to purchase a specific quantity at the lowest price.
  
  **Seller's Goal**
  
  The seller wants to sell a specific quantity at the highest price.

issue_description:
  Quantity: The number of units to trade
  Price: The price per unit

hints:
  Buyer:
    Target Quantity: 5
    Price Sensitivity: High
  Seller:
    Target Quantity: 8
    Price Sensitivity: Medium
```

## Creating a New Scenario

### Step 1: Create Directory

```bash
mkdir -p ~/negmas/hani/settings/scenarios/MyCategory/MyScenario
```

### Step 2: Define Outcome Space

Create `MyScenario.yml`:

```yaml
name: MyScenario
type: DiscreteCartesianOutcomeSpace
issues:
  - name: IssueA
    type: ContiguousIssue
    values: [1, 10]
  - name: IssueB
    type: DiscreteCardinalIssue
    values: [100, 200, 300]
```

### Step 3: Define Human Utility

Create `human.yml`:

```yaml
name: human
type: LinearAdditiveUtilityFunction
reserved_value: 0.1
weights: [0.6, 0.4]
values:
  - type: LinearFun
    min_value: 0.0
    max_value: 1.0
  - type: TableFun
    mapping:
      100: 0.0
      200: 0.5
      300: 1.0
```

### Step 4: Define Agent Utility

Create `agent.yml`:

```yaml
name: agent
type: LinearAdditiveUtilityFunction
reserved_value: 0.1
weights: [0.4, 0.6]
values:
  - type: LinearFun
    min_value: 1.0
    max_value: 0.0
  - type: TableFun
    mapping:
      100: 1.0
      200: 0.5
      300: 0.0
```

### Step 5: Add Metadata (Optional)

Create `_info.yaml`:

```yaml
Name: MyScenario
title: My Custom Scenario
short_description: A brief description of the scenario.
issue_description:
  IssueA: Description of issue A
  IssueB: Description of issue B
```

### Step 6: Test

Restart HANI and your scenario should appear in the scenario selector.

## Built-in Scenarios

HANI includes several default scenarios:

| Scenario | Description |
|----------|-------------|
| Trade | Buyer/seller negotiating quantity and price |
| Grocery | Shopping negotiation with multiple items |
| Island | Resource allocation on a deserted island |

Run `hani setup` to copy these to your settings directory as examples.

## Tips

!!! tip "Balanced Scenarios"
    Good scenarios have:
    
    - Conflicting but not completely opposed preferences
    - Room for mutual gain (win-win outcomes exist)
    - Meaningful trade-offs between issues

!!! tip "Testing Utilities"
    Verify your utility functions:
    
    - Weights should sum to 1.0
    - Values should be normalized (0.0 to 1.0)
    - Reserved value should be achievable (not too high)
