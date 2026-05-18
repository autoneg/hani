from negmas.preferences.value_fun import TableFun
from negmas.helpers.inout import load
from negmas import (
    DiscreteCartesianOutcomeSpace,
    LinearAdditiveUtilityFunction,
    UtilityFunction,
    make_issue,
    make_os,
    Outcome,
    Scenario,
)
from hani.common import DEFAULT_SCENRIOS, DefaultOutcomeDisplay, INFO_FILE_NAME
from .common import IntRange, float_in, int_in, FloatRange


Compass = "Compass 🧭"
Container = "Container 🧰"
Food = "Food 🍲"
Hammer = "Hammer 🔨"
Knife = "Knife 🔪"
Match = "Match 🔥"
Medicine = "Medicine 💊"
Rope = "Rope 🪢"


ICON = {
    Compass: "🧭",
    Container: "🧰",
    Food: "🍲",
    Hammer: "🔨",
    Knife: "🔪",
    Match: "🔥",
    Medicine: "💊",
    Rope: "🪢",
}


def make_island_scenario(
    index: int,
    compass_agent_value: IntRange = (3, 25),
    container_agent_value: IntRange = (3, 25),
    food_agent_value: IntRange = (3, 25),
    hammer_agent_value: IntRange = (3, 25),
    knife_agent_value: IntRange = (3, 25),
    match_agent_value: IntRange = (3, 25),
    medicine_agent_value: IntRange = (3, 25),
    rope_agent_value: IntRange = (3, 25),
    compass_human_value: IntRange = (1, 22),
    container_human_value: IntRange = (1, 22),
    food_human_value: IntRange = (1, 22),
    hammer_human_value: IntRange = (1, 22),
    knife_human_value: IntRange = (1, 22),
    match_human_value: IntRange = (1, 22),
    medicine_human_value: IntRange = (1, 22),
    rope_human_value: IntRange = (1, 22),
    agent_reserved_range: FloatRange = (0.0, 0.3),
    human_reserved_range: FloatRange = (0.0, 0.3),
    normalize: bool = True,
    offer_for_agent: bool = False,
) -> Scenario:
    vals = dict(
        agent={
            Compass: int_in(compass_agent_value),
            Container: int_in(container_agent_value),
            Food: int_in(food_agent_value),
            Hammer: int_in(hammer_agent_value),
            Knife: int_in(knife_agent_value),
            Match: int_in(match_agent_value),
            Medicine: int_in(medicine_agent_value),
            Rope: int_in(rope_agent_value),
        },
        human={
            Compass: int_in(compass_human_value),
            Container: int_in(container_human_value),
            Food: int_in(food_human_value),
            Hammer: int_in(hammer_human_value),
            Knife: int_in(knife_human_value),
            Match: int_in(match_human_value),
            Medicine: int_in(medicine_human_value),
            Rope: int_in(rope_human_value),
        },
    )
    mn, mx = float("inf"), float("-inf")
    best = dict(agent="", human="")
    worst = dict(agent="", human="")
    for person in vals:
        for item in vals[person]:
            mn = min(mn, vals[person][item])
            mx = max(mx, vals[person][item])
            if not best[person] or vals[person][item] > vals[person][best[person]]:
                best[person] = item
            if not worst[person] or vals[person][item] < vals[person][worst[person]]:
                worst[person] = item

    os = make_os(
        [
            make_issue(["agent", "human"], name=Compass),
            make_issue(["agent", "human"], name=Container),
            make_issue(["agent", "human"], name=Food),
            make_issue(["agent", "human"], name=Hammer),
            make_issue(["agent", "human"], name=Knife),
            make_issue(["agent", "human"], name=Match),
            make_issue(["agent", "human"], name=Medicine),
            make_issue(["agent", "human"], name=Rope),
        ],
        name="Island",
    )
    assert isinstance(os, DiscreteCartesianOutcomeSpace)

    def make_ufun(target: str) -> UtilityFunction:
        weights = [
            vals[target][item.name]
            if not normalize
            else ((vals[target][item.name] - mn + 1) / (mx - mn + 1))
            for item in os.issues
        ]
        s = sum(weights)
        weights = [_ / s for _ in weights]
        return LinearAdditiveUtilityFunction(
            tuple(
                [
                    TableFun(
                        dict(agent=int(target == "agent"), human=int(target != "agent"))
                    )
                    for _ in os.issues
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

    info = load(DEFAULT_SCENRIOS / "Island" / INFO_FILE_NAME)
    info["hints"]["agent"]["Most Important Item"] = best["agent"]
    info["hints"]["agent"]["Least Important Item"] = worst["agent"]
    info["hints"]["human"]["Most Important Item"] = best["human"]
    info["hints"]["human"]["Least Important Item"] = worst["human"]
    return Scenario(outcome_space=os, ufuns=tuple(ufuns), info=info)


class IslandOutcomeDisplay(DefaultOutcomeDisplay):
    def __init__(self, *args, for_agent=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._for_agent = for_agent

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
        items = sorted(
            [
                issues[i].name
                for i, item in enumerate(outcome)
                if (item == "agent" and self._for_agent)
                or (item != "agent" and not self._for_agent)
            ]
        )
        target = "I" if from_human else "You"
        if not items:
            return f"{target} get nothing."
        if len(items) == len(outcome):
            return f"{target} get everything."
        return f"{target} get {', '.join([ICON.get(_.lower(), _) for _ in items])}."
