import itertools
from negmas.helpers.inout import load
from negmas import (
    AffineFun,
    DiscreteCartesianOutcomeSpace,
    LinearAdditiveUtilityFunction,
    LinearFun,
    UtilityFunction,
    make_issue,
    make_os,
    Outcome,
    Scenario,
)
from hani.common import DEFAULT_SCENRIOS, DefaultOutcomeDisplay, INFO_FILE_NAME
from .common import IntRange, float_in, int_in, FloatRange

Agent = "agent"
Human = "human"
Watermelon = "watermelon 🍉"
Banana = "banana 🍌"
Orange = "orange 🟠"
Apple = "apple 🍎"
ICONS = {Watermelon: "🍉", Banana: "🍌", Orange: "🟠", Apple: "🍎"}
N_PER_ITEM = 4


def make_grocery_scenario(
    index: int,
    watermelon_agent_value: IntRange = (3, 25),
    banana_agent_value: IntRange = (3, 25),
    orange_agent_value: IntRange = (3, 25),
    apple_agent_value: IntRange = (3, 25),
    watermelon_human_value: IntRange = (1, 22),
    banana_human_value: IntRange = (1, 22),
    orange_human_value: IntRange = (1, 22),
    apple_human_value: IntRange = (1, 22),
    n_per_item: IntRange = N_PER_ITEM,
    agent_reserved_range: FloatRange = (0.0, 0.3),
    human_reserved_range: FloatRange = (0.0, 0.3),
    normalize: bool = True,
    offer_for_agent: bool = False,
) -> Scenario:
    N = int_in(n_per_item)
    vals = dict(
        agent={
            Watermelon: int_in(watermelon_agent_value),
            Banana: int_in(banana_agent_value),
            Orange: int_in(orange_agent_value),
            Apple: int_in(apple_agent_value),
        },
        human={
            Watermelon: int_in(watermelon_human_value),
            Banana: int_in(banana_human_value),
            Orange: int_in(orange_human_value),
            Apple: int_in(apple_human_value),
        },
    )
    mn, mx = 0.0, float("-inf")
    best = dict(agent="", human="")
    worst = dict(agent="", human="")
    best_val = dict(
        agent=dict(zip(vals["agent"].keys(), itertools.repeat(-1))),
        human=dict(zip(vals["human"].keys(), itertools.repeat(-1))),
    )
    for person in vals:
        for item in vals[person]:
            v = vals[person][item]
            mx = max(mx, v)
            if not best[person] or v > vals[person][best[person]]:
                best[person] = item
            if not best_val[person][item] < 0 or v >= best_val[person][item]:
                best_val[person][item] = v
            if not worst[person] or v < vals[person][worst[person]]:
                worst[person] = item
    os = make_os(
        [
            make_issue(N + 1, name=Apple),
            make_issue(N + 1, name=Banana),
            make_issue(N + 1, name=Orange),
            make_issue(N + 1, name=Watermelon),
        ],
        name="Grocery",
    )
    assert isinstance(os, DiscreteCartesianOutcomeSpace)

    def make_ufun(target: str) -> UtilityFunction:
        offer_for = "agent" if offer_for_agent else "human"
        if target == offer_for:
            weights = [
                ((vals[target][_.name]) / mx) if normalize else vals[target][_.name]
                for _ in os.issues
            ]
            if normalize:
                s = sum(weights)
                weights = [w / s for w in weights]
            slopes = [1 / N] * len(os.issues) if normalize else ([1.0] * len(os.issues))
            biases = [0.0] * len(os.issues)
        else:
            weights = [
                ((vals[target][_.name]) / mx) if normalize else vals[target][_.name]
                for _ in os.issues
            ]
            if normalize:
                s = sum(weights)
                weights = [w / s for w in weights]
            slopes = (
                [-1 / N] * len(os.issues) if normalize else ([-1.0] * len(os.issues))
            )
            biases = [
                1.0 if normalize else (best_val[target][_.name] - vals[target][_.name])
                for _ in os.issues
            ]
        return LinearAdditiveUtilityFunction(
            tuple(
                [
                    AffineFun(slope=s, bias=b) if b else LinearFun(slope=s)
                    for s, b in zip(slopes, biases)
                ]
            ),
            weights=tuple(weights),
            reserved_value=float_in(agent_reserved_range)
            if target == "agent"
            else float_in(human_reserved_range),
            name=target,
            id=target,
            outcome_space=os,
        )

    ufuns = [make_ufun("agent"), make_ufun("human")]

    info = load(DEFAULT_SCENRIOS / "Grocery" / INFO_FILE_NAME)
    info["hints"]["agent"]["Most Important Item"] = best["agent"]
    info["hints"]["agent"]["Least Important Item"] = worst["agent"]
    info["hints"]["human"]["Most Important Item"] = best["human"]
    info["hints"]["human"]["Least Important Item"] = worst["human"]
    scenario = Scenario(outcome_space=os, ufuns=tuple(ufuns), info=info)
    return scenario


class GroceryOutcomeDisplay(DefaultOutcomeDisplay):
    def __init__(self, *args, for_agent=False, n_per_item=N_PER_ITEM, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._for_agent = for_agent
        self._n_per_item = n_per_item

    def str(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> str:
        if outcome is None:
            return super().str(outcome, scenario, is_done, from_human)
        issues = scenario.outcome_space.issues
        counts = [
            int(item) if from_human else int(4 - item) for i, item in enumerate(outcome)
        ]
        items = [
            f"{n} {ICONS.get(issues[i].name.lower(), issues[i].name + ('s' if item > 1 else ''))}"
            for i, (item, n) in enumerate(zip(counts, outcome))
            if n
        ]
        if not items:
            return f"{'I' if from_human else 'You'} get nothing."
        if sum(counts) == (self._n_per_item) * len(outcome):
            return f"{'I' if from_human else 'You'} get everything."
        return f"{'I' if from_human else 'You'} get {', '.join([_ for _ in items])}."
