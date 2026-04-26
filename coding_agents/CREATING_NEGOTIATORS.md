# Creating Custom Negotiators for HANI

**Complete Guide for AI Negotiator Development**

This guide explains how to create custom automated negotiators (AI agents) for the HANI (Human Agent Negotiation Interface). Whether you're implementing classic negotiation strategies or developing novel AI approaches, this document will guide you through the process.

---

## Table of Contents

1. [Understanding Negotiators in HANI](#understanding-negotiators-in-hani)
2. [Negotiator Architecture](#negotiator-architecture)
3. [Step-by-Step: Creating Your First Negotiator](#step-by-step-creating-your-first-negotiator)
4. [Advanced Negotiator Examples](#advanced-negotiator-examples)
5. [Opponent Modeling](#opponent-modeling)
6. [Testing Your Negotiator](#testing-your-negotiator)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Understanding Negotiators in HANI

### What is a Negotiator?

In HANI, a **Negotiator** is an AI agent that:
- Makes offers during bilateral negotiations
- Responds to opponent offers (accept, reject, or end)
- Has a utility function representing its preferences
- Can learn about the opponent during negotiation
- Operates under time and step constraints

### Negotiation Protocol

HANI uses the **Stacked Alternating Offers (SAO)** protocol:
1. Negotiators take turns making offers
2. An offer can be: a specific outcome from the outcome space
3. Responses can be: ACCEPT_OFFER, REJECT_OFFER (with counter-offer), or END_NEGOTIATION
4. Negotiation ends when: agreement reached, time runs out, steps exhausted, or a party ends it

### Built-in Negotiators

HANI includes several negotiators from competition winners:

| Negotiator | Strategy | Source |
|------------|----------|--------|
| **HybridNegotiator** | Aspiration-based with time pressure | NegMAS default |
| **AverageTitForTat** | Reciprocal concession based on average | ANAC competition |
| **HardHeaded** | Stubborn with limited concessions | ANAC competition |
| **AgentK** | Bayesian opponent modeling | ANAC competition |
| **Atlas3** | Complex bidding and acceptance strategy | ANAC 2012 winner |
| **CUHKAgent** | Time-dependent concession | ANAC competition |
| **AgentGG** | Genetic algorithm-based | ANAC competition |

---

## Negotiator Architecture

### Base Class: SAONegotiator

All HANI negotiators inherit from `SAONegotiator`:

```python
from negmas.sao import SAONegotiator, SAOState, SAOResponse
from negmas import ResponseType, Outcome

class MyNegotiator(SAONegotiator):
    """My custom negotiator"""
    
    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Called when it's this negotiator's turn to act.
        
        Args:
            state: Current negotiation state (step, time, current offer, etc.)
            
        Returns:
            SAOResponse with response type and optional counter-offer
        """
        # Your negotiation logic here
        pass
```

### Key Components

#### 1. Utility Function (`self.ufun`)
```python
# Get utility of an outcome
utility = self.ufun(outcome)  # Returns float in [0, 1]

# Get best outcome
best_outcome = self.ufun.best()

# Get reserved value (minimum acceptable utility)
reservation = self.ufun.reserved_value
```

#### 2. Negotiation State (`state: SAOState`)
```python
state.step              # Current negotiation step
state.time              # Current time elapsed
state.relative_time     # Progress: 0.0 to 1.0
state.current_offer     # Last offer from opponent
state.current_proposer  # Who made current offer
state.n_negotiators     # Number of negotiators (usually 2)
```

#### 3. Mechanism Interface (`self.nmi`)
```python
self.nmi.n_steps          # Maximum steps allowed
self.nmi.time_limit       # Maximum time allowed
self.nmi.outcome_space    # All possible outcomes
self.nmi.issues           # List of negotiation issues
```

#### 4. Response Types
```python
ResponseType.ACCEPT_OFFER      # Accept current offer
ResponseType.REJECT_OFFER      # Reject and make counter-offer
ResponseType.END_NEGOTIATION   # End negotiation (failure)
```

---

## Step-by-Step: Creating Your First Negotiator

### Example 1: Random Negotiator (Simplest)

```python
from negmas.sao import SAONegotiator, SAOState, SAOResponse
from negmas import ResponseType

class RandomNegotiator(SAONegotiator):
    """
    Accepts with 50% probability, otherwise makes random counter-offer.
    """
    
    def __call__(self, state: SAOState) -> SAOResponse:
        import random
        
        # 50% chance to accept current offer
        if random.random() < 0.5 and state.current_offer is not None:
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        
        # Otherwise, make a random counter-offer
        random_outcome = self.nmi.random_outcome()
        return SAOResponse(ResponseType.REJECT_OFFER, random_outcome)
```

### Example 2: Time-Based Concession Negotiator

```python
class TimeConcessionNegotiator(SAONegotiator):
    """
    Starts with best outcome, gradually concedes based on time.
    Accepts offers above current aspiration level.
    """
    
    def __call__(self, state: SAOState) -> SAOResponse:
        # Calculate current aspiration level (decreases over time)
        # At t=0: aspiration=1.0, At t=1: aspiration=reserved_value
        aspiration = (
            self.ufun.reserved_value + 
            (1.0 - self.ufun.reserved_value) * (1 - state.relative_time)
        )
        
        # Accept if opponent's offer meets aspiration
        current_offer = state.current_offer
        if current_offer is not None:
            utility = self.ufun(current_offer)
            if utility >= aspiration:
                return SAOResponse(ResponseType.ACCEPT_OFFER, current_offer)
        
        # Make counter-offer at aspiration level
        # Find outcome closest to aspiration
        my_offer = self._find_outcome_near_utility(aspiration)
        return SAOResponse(ResponseType.REJECT_OFFER, my_offer)
    
    def _find_outcome_near_utility(self, target_utility: float) -> Outcome:
        """Find outcome with utility close to target."""
        # Get all outcomes
        outcomes = list(self.nmi.discrete_outcomes())
        
        # Find closest to target
        best_outcome = outcomes[0]
        best_diff = abs(self.ufun(best_outcome) - target_utility)
        
        for outcome in outcomes[1:]:
            diff = abs(self.ufun(outcome) - target_utility)
            if diff < best_diff:
                best_diff = diff
                best_outcome = outcome
        
        return best_outcome
```

### Example 3: Boulware Strategy (Tough Negotiator)

```python
class BoulwareNegotiator(SAONegotiator):
    """
    Concedes very slowly, staying near best outcome until late in negotiation.
    Uses exponential concession curve.
    """
    
    def __call__(self, state: SAOState) -> SAOResponse:
        # Boulware concession: slow at first, faster near deadline
        # Formula: utility = reserved + (1 - reserved) * (1 - t^5)
        reserved = self.ufun.reserved_value
        concession_rate = 5.0  # Higher = more stubborn
        
        aspiration = reserved + (1 - reserved) * (1 - state.relative_time ** concession_rate)
        
        # Accept good offers
        if state.current_offer is not None:
            utility = self.ufun(state.current_offer)
            if utility >= aspiration:
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        
        # Make tough counter-offer
        my_offer = self._find_outcome_near_utility(aspiration)
        return SAOResponse(ResponseType.REJECT_OFFER, my_offer)
    
    def _find_outcome_near_utility(self, target_utility: float) -> Outcome:
        """Find outcome with utility close to target."""
        # Use utility inverter if available
        if hasattr(self.ufun, 'invert'):
            inverter = self.ufun.invert()
            outcome = inverter.one_in((target_utility, target_utility + 0.05))
            if outcome:
                return outcome
        
        # Fallback: search through outcomes
        outcomes = list(self.nmi.discrete_outcomes())
        return min(outcomes, key=lambda o: abs(self.ufun(o) - target_utility))
```

---

## Advanced Negotiator Examples

### Example 4: Tit-for-Tat Negotiator

```python
class TitForTatNegotiator(SAONegotiator):
    """
    Reciprocates opponent's concession behavior.
    If opponent concedes, we concede. If opponent is tough, we're tough.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opponent_utilities = []  # Track opponent's offers
        self.my_last_utility = None
    
    def __call__(self, state: SAOState) -> SAOResponse:
        # Track opponent's utility for their offer
        if state.current_offer is not None:
            opp_utility = self.ufun(state.current_offer)
            self.opponent_utilities.append(opp_utility)
        
        # Calculate opponent's average concession rate
        if len(self.opponent_utilities) >= 2:
            recent_offers = self.opponent_utilities[-5:]  # Last 5 offers
            opponent_concession = sum(recent_offers) / len(recent_offers)
        else:
            opponent_concession = 0.5  # Start neutral
        
        # Accept if offer is good
        if state.current_offer and self.ufun(state.current_offer) >= opponent_concession:
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        
        # Mirror opponent's concession behavior
        # If they're conceding (higher utilities), we concede too
        if self.my_last_utility is None:
            target_utility = 0.9  # Start high
        else:
            # Concede at same rate as opponent
            if len(self.opponent_utilities) >= 2:
                opp_change = self.opponent_utilities[-1] - self.opponent_utilities[-2]
                target_utility = max(
                    self.ufun.reserved_value,
                    self.my_last_utility - abs(opp_change)
                )
            else:
                target_utility = self.my_last_utility * 0.95
        
        my_offer = self._find_outcome_near_utility(target_utility)
        self.my_last_utility = self.ufun(my_offer)
        
        return SAOResponse(ResponseType.REJECT_OFFER, my_offer)
    
    def _find_outcome_near_utility(self, target: float) -> Outcome:
        outcomes = list(self.nmi.discrete_outcomes())
        return min(outcomes, key=lambda o: abs(self.ufun(o) - target))
```

### Example 5: Negotiator with Opponent Modeling

```python
from collections import defaultdict

class OpponentModelingNegotiator(SAONegotiator):
    """
    Learns opponent's preferences by observing their offers.
    Uses frequency-based modeling to estimate issue weights.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opponent_offers = []
        self.issue_frequencies = defaultdict(lambda: defaultdict(int))
    
    def __call__(self, state: SAOState) -> SAOResponse:
        # Learn from opponent's offer
        if state.current_offer is not None:
            self._learn_from_offer(state.current_offer)
            
            # Accept if offer is good enough
            my_utility = self.ufun(state.current_offer)
            estimated_opp_utility = self._estimate_opponent_utility(state.current_offer)
            
            # Accept if Nash-like equilibrium
            if my_utility >= 0.7 and estimated_opp_utility >= 0.7:
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        
        # Find win-win offer using opponent model
        my_offer = self._find_win_win_offer(state)
        return SAOResponse(ResponseType.REJECT_OFFER, my_offer)
    
    def _learn_from_offer(self, offer: Outcome):
        """Learn opponent preferences from their offer."""
        self.opponent_offers.append(offer)
        
        # Count value frequencies for each issue
        for i, value in enumerate(offer):
            issue_name = self.nmi.issues[i].name
            self.issue_frequencies[issue_name][value] += 1
    
    def _estimate_opponent_utility(self, offer: Outcome) -> float:
        """Estimate opponent's utility for an offer."""
        if len(self.opponent_offers) < 3:
            return 0.5  # Not enough data
        
        # Simple frequency-based estimate
        score = 0.0
        for i, value in enumerate(offer):
            issue_name = self.nmi.issues[i].name
            freq = self.issue_frequencies[issue_name][value]
            total_freq = sum(self.issue_frequencies[issue_name].values())
            score += freq / total_freq if total_freq > 0 else 0
        
        return score / len(offer)
    
    def _find_win_win_offer(self, state: SAOState) -> Outcome:
        """Find outcome that's good for both parties."""
        # Calculate current aspiration
        aspiration = self.ufun.reserved_value + (1 - self.ufun.reserved_value) * (1 - state.relative_time ** 2)
        
        # Find outcomes good for us
        outcomes = list(self.nmi.discrete_outcomes())
        good_outcomes = [o for o in outcomes if self.ufun(o) >= aspiration]
        
        if not good_outcomes:
            good_outcomes = outcomes
        
        # Among good outcomes, pick one that's likely good for opponent
        best_offer = max(
            good_outcomes,
            key=lambda o: self._estimate_opponent_utility(o)
        )
        
        return best_offer
```

---

## Opponent Modeling

### Why Model the Opponent?

Opponent modeling helps you:
- Predict which offers they'll accept
- Find mutually beneficial outcomes
- Avoid wasting time on unacceptable offers
- Negotiate more efficiently

### Modeling Techniques

#### 1. Frequency-Based Modeling

```python
class FrequencyModel:
    """Learn preferences by tracking value frequencies."""
    
    def __init__(self):
        self.value_counts = defaultdict(lambda: defaultdict(int))
    
    def update(self, offer: Outcome, issues: list):
        """Record an offer."""
        for i, value in enumerate(offer):
            issue_name = issues[i].name
            self.value_counts[issue_name][value] += 1
    
    def estimate_utility(self, offer: Outcome, issues: list) -> float:
        """Estimate utility based on value frequencies."""
        if not self.value_counts:
            return 0.5
        
        scores = []
        for i, value in enumerate(offer):
            issue_name = issues[i].name
            freq = self.value_counts[issue_name][value]
            total = sum(self.value_counts[issue_name].values())
            scores.append(freq / total if total > 0 else 0.5)
        
        return sum(scores) / len(scores)
```

#### 2. Bayesian Learning

```python
class BayesianModel:
    """Use Bayesian updates to learn utility function."""
    
    def __init__(self, n_hypotheses=10):
        # Generate hypothesis utility functions
        self.hypotheses = self._generate_hypotheses(n_hypotheses)
        self.weights = [1.0 / n_hypotheses] * n_hypotheses
    
    def update(self, accepted_offers: list, rejected_offers: list):
        """Update beliefs based on accept/reject decisions."""
        for i, hypothesis in enumerate(self.hypotheses):
            # Likelihood: hypothesis consistent with observations?
            likelihood = 1.0
            
            for offer in accepted_offers:
                # Hypothesis should give high utility to accepted offers
                utility = hypothesis(offer)
                likelihood *= utility
            
            for offer in rejected_offers:
                # Hypothesis should give low utility to rejected offers
                utility = hypothesis(offer)
                likelihood *= (1 - utility)
            
            self.weights[i] *= likelihood
        
        # Normalize weights
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]
    
    def estimate_utility(self, offer: Outcome) -> float:
        """Estimate utility as weighted average of hypotheses."""
        return sum(
            w * h(offer)
            for w, h in zip(self.weights, self.hypotheses)
        )
    
    def _generate_hypotheses(self, n: int):
        """Generate random utility function hypotheses."""
        # Implementation depends on your needs
        pass
```

---

## Testing Your Negotiator

### 1. Unit Testing

```python
# test_my_negotiator.py
import pytest
from negmas import SAOMechanism
from negmas.preferences import LinearAdditiveUtilityFunction
from my_negotiator import MyNegotiator

def test_negotiator_basic():
    """Test negotiator makes valid responses."""
    # Create simple scenario
    from negmas import make_issue
    
    issues = [make_issue([1, 2, 3], "price")]
    n_steps = 10
    
    # Create mechanism
    mech = SAOMechanism(issues=issues, n_steps=n_steps)
    
    # Create negotiators
    ufun1 = LinearAdditiveUtilityFunction(...)
    ufun2 = LinearAdditiveUtilityFunction(...)
    
    neg1 = MyNegotiator(ufun=ufun1)
    neg2 = MyNegotiator(ufun=ufun2)
    
    mech.add(neg1)
    mech.add(neg2)
    
    # Run negotiation
    mech.run()
    
    # Check result
    assert mech.state.agreement is not None or mech.state.step == n_steps
```

### 2. Interactive Testing in HANI

Add your negotiator to HANI for human testing:

**Step 1**: Add to HANI helpers

```python
# src/hani/helpers/negotiators.py
from my_negotiator import MyNegotiator

__all__ = [
    # ... existing negotiators
    "MyNegotiator",
]
```

**Step 2**: Register in app.py

```python
# src/hani/app.py
HANI_NEGOTIATORS = [
    "helpers.AverageTitForTat",
    "helpers.HardHeaded",
    # ... other negotiators
    "helpers.MyNegotiator",  # Add yours
]
```

**Step 3**: Test against humans

```bash
hani --dev
# Login, select scenario, negotiate against your AI
```

### 3. Tournament Testing

```python
# Run tournament between negotiators
from negmas.situated import tournament

results = tournament(
    competitors=[MyNegotiator, AverageTitForTat, HardHeaded],
    scenarios=["Trade", "Island", "Grocery"],
    n_repetitions=10,
    n_steps=100,
)

print(results.scores)
```

---

## Best Practices

### 1. Algorithm Design

**Do**:
- Start simple, add complexity gradually
- Handle edge cases (first step, last step, no current offer)
- Use aspiration levels for accept/reject decisions
- Consider both your utility AND opponent's likely utility

**Don't**:
- Assume utilities are zero-sum (they often aren't!)
- Accept offers below your reservation value
- Make offers worse than previous ones (non-monotonic)
- Ignore time/step constraints

### 2. Performance

```python
# ✅ GOOD: Cache expensive computations
class EfficientNegotiator(SAONegotiator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sorted_outcomes = None  # Cache
    
    def _get_sorted_outcomes(self):
        if self._sorted_outcomes is None:
            outcomes = list(self.nmi.discrete_outcomes())
            self._sorted_outcomes = sorted(
                outcomes,
                key=lambda o: self.ufun(o),
                reverse=True
            )
        return self._sorted_outcomes

# ❌ BAD: Recompute every time
class SlowNegotiator(SAONegotiator):
    def __call__(self, state):
        # Recomputes sorting every call!
        outcomes = sorted(
            self.nmi.discrete_outcomes(),
            key=self.ufun,
            reverse=True
        )
```

### 3. Utility Function Usage

```python
# ✅ GOOD: Check utility before accepting
if state.current_offer and self.ufun(state.current_offer) >= threshold:
    return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

# ❌ BAD: Accept without checking
if state.relative_time > 0.9:
    return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
```

### 4. Handling Edge Cases

```python
def __call__(self, state: SAOState) -> SAOResponse:
    # ✅ GOOD: Handle None offer (first step)
    if state.current_offer is None:
        # Make initial offer
        return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
    
    # ❌ BAD: Assume offer exists
    utility = self.ufun(state.current_offer)  # May crash on first step!
```

---

## Troubleshooting

### Negotiator Doesn't Appear in HANI

**Problem**: Added negotiator but it doesn't show in partner list.

**Solutions**:
1. Check it's added to `__all__` in negotiators.py
2. Verify it's registered in `HANI_NEGOTIATORS` in app.py
3. Restart HANI server
4. Check for Python import errors in terminal

### Negotiation Always Fails

**Problem**: Never reaches agreement.

**Solutions**:
1. Check acceptance threshold isn't too high
2. Verify concession happens over time
3. Check if offers are getting stuck at same value
4. Log utilities to see if they're reasonable

### Performance Issues

**Problem**: Negotiator is too slow.

**Solutions**:
1. Cache expensive computations (outcome sorting, inverters)
2. Limit opponent model complexity
3. Use sampling instead of exhaustive search
4. Profile code to find bottlenecks

### Unexpected Behavior

**Problem**: Negotiator acts strangely.

**Solutions**:
1. Add logging: `print(f"Step {state.step}: utility={utility}, aspiration={aspiration}")`
2. Check state.relative_time is in [0, 1]
3. Verify ufun returns values in [0, 1]
4. Test with simple scenarios first

---

## Complete Checklist for Creating a Negotiator

- [ ] Inherit from `SAONegotiator`
- [ ] Implement `__call__(self, state: SAOState) -> SAOResponse`
- [ ] Handle None current_offer (first step)
- [ ] Check utilities before accepting
- [ ] Implement concession strategy (time-based, opponent-based, etc.)
- [ ] Respect reserved value (never accept below it)
- [ ] Return valid SAOResponse with ResponseType and offer
- [ ] Add docstring explaining strategy
- [ ] Add to HANI_NEGOTIATORS list
- [ ] Test in unit tests
- [ ] Test against humans in HANI
- [ ] Test in tournaments against other negotiators

---

## Additional Resources

- **NegMAS Documentation**: https://negmas.readthedocs.io/
- **SAO Protocol**: https://negmas.readthedocs.io/en/latest/api/negmas.sao.html
- **ANAC Competition**: http://web.tuat.ac.jp/~katfuji/ANAC/
- **HANI Negotiator Examples**: `src/hani/helpers/negotiators.py`

---

## Summary

Creating a negotiator involves:

1. **Inherit** from `SAONegotiator`
2. **Implement** the `__call__` method with your strategy
3. **Use** utility function to evaluate offers
4. **Consider** time pressure and opponent behavior
5. **Test** thoroughly in different scenarios
6. **Optimize** for performance
7. **Register** in HANI for human testing

The key to a good negotiator is balancing ambition (getting high utility) with pragmatism (making deals happen). Start simple, test often, and iterate based on performance!

Good luck building your negotiator! 🤖🤝
