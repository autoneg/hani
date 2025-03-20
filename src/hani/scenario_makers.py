import random
from negmas.helpers.inout import load
from numpy import argmin
from typing import Iterable
from negmas import (
    DiscreteCartesianOutcomeSpace,
    LinearAdditiveUtilityFunction,
    AffineFun,
    Scenario,
    TableFun,
    make_issue,
    make_os,
)
from hani.common import SAMPLE_SCENRIOS


__all__ = ["make_trade_scenario", "make_colored_chips"]

FloatRange = tuple[float, float] | float
IntRange = tuple[int, int] | list[int] | int


def float_in(x: FloatRange):
    if isinstance(x, Iterable):
        return x[0] + (x[1] - x[0]) * random.random()
    return x


def int_in(x: IntRange):
    if isinstance(x, Iterable):
        return random.randint(min(x), max(x))
    return x


def make_values(
    qs: list[int], target: int, shortfall: FloatRange, excess: FloatRange
) -> dict[int, float]:
    target = int_in(target)
    if target not in qs:
        dists = [abs(_ - target) for _ in qs]
        target = int(argmin(dists))
    else:
        target = qs.index(target)
    if target > max(qs):
        target = max(qs)
    if target < min(qs):
        target = min(qs)
    values = [0.0] * len(qs)
    values[target] = 1.0
    for i in range(target - 1, -1, -1):
        values[i] = max(0.0, values[i + 1] - float_in(shortfall))
    for i in range(target + 1, len(values)):
        values[i] = max(0.0, values[i - 1] - float_in(excess))
    return dict(zip(qs, values))


def make_trade_scenario(
    index: int,
    quantity: IntRange = (1, 20),
    price: tuple[int, int] | list[int] = list(range(100, 200, 20)),
    seller_name: str = "Seller",
    seller_target_quantity: IntRange | None = None,
    seller_reserved_value: FloatRange = (0.0, 0.4),
    seller_shortfall_penalty: FloatRange = (0.1, 0.3),
    seller_excess_penalty: FloatRange = (0.1, 0.4),
    seller_quantity_weight: FloatRange = (0.1, 0.9),
    buyer_name: str = "Buyer",
    buyer_target_quantity: IntRange | None = None,
    buyer_reserved_value: FloatRange = (0.0, 0.4),
    buyer_shortfall_penalty: FloatRange = (0.1, 0.3),
    buyer_excess_penalty: FloatRange = (0.1, 0.4),
    buyer_quantity_weight: FloatRange = (0.1, 0.9),
) -> Scenario:
    """Creates a target-quantity type scenario"""
    buyer_params = dict(
        name=buyer_name,
        target=buyer_target_quantity,
        reserved=buyer_reserved_value,
        shortfall=buyer_shortfall_penalty,
        excess=buyer_excess_penalty,
        qweight=buyer_quantity_weight,
    )
    seller_params = dict(
        name=seller_name,
        target=seller_target_quantity,
        reserved=seller_reserved_value,
        shortfall=seller_shortfall_penalty,
        excess=seller_excess_penalty,
        qweight=seller_quantity_weight,
    )

    params = [seller_params, buyer_params]
    names = ["Seller", "Buyer"]
    for d in params:
        if d["target"] is None:
            d["target"] = quantity
    os = make_os(
        [make_issue(quantity, name="Quantity"), make_issue(price, name="Price")],
        name="Trade",
    )
    assert isinstance(os, DiscreteCartesianOutcomeSpace)
    quantities = list(os.issues[0].all)
    ufuns = []
    for name, p in zip(names, params):
        w = float_in(p["qweight"])  # type: ignore
        mn, mx = os.issues[1].min_value, os.issues[1].max_value
        d = mx - mn
        ufun = LinearAdditiveUtilityFunction(
            values=(
                TableFun(
                    make_values(quantities, p["target"], p["shortfall"], p["excess"])  # type: ignore
                ),
                AffineFun(slope=1 / d, bias=-mn / d),
            ),
            weights=(w, 1.0 - w),
            outcome_space=os,
            reserved_value=float_in(p["reserved"]),  # type: ignore
            name=name,
            id=name,
        )
        ufuns.append(ufun)

    return Scenario(
        outcome_space=os,
        ufuns=tuple(ufuns),
        info=load(SAMPLE_SCENRIOS / "Trade" / "_info.yaml"),
    )


def make_colored_chips(index: int):
    return make_trade_scenario(index, (1, 5))
