import math
import random
from typing import Iterable
import numpy as np
from negmas.preferences.value_fun import TableFun, AffineFun, LinearFun, LambdaFun
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
from hani.common import SAMPLE_SCENRIOS, DefaultOutcomeDisplay, INFO_FILE_NAME

__ = TableFun, AffineFun, LinearFun, LambdaFun

FloatRange = tuple[float, float] | tuple[float, float, int] | list[float] | float
IntRange = tuple[int, int] | tuple[int, int, int] | list[int] | int
FloatIssueRange = tuple[float, float] | tuple[float, float, float] | list[float]

# random.seed(1234)
# np.random.seed(1234)


def range_in(x: FloatIssueRange):
    if isinstance(x, tuple) and len(x) == 3:
        x = np.round(np.linspace(x[0], x[1], num=x[2], endpoint=True)).tolist()  # type: ignore
    if isinstance(x, tuple) and len(x) == 2:
        return range_in((*x, 11))  # type: ignore
    if isinstance(x, list):
        return x
    raise ValueError(f"Unsupported iterable {x}")


def float_in(x: FloatRange):
    if isinstance(x, tuple) and len(x) == 3:
        num = (x[1] - x[0]) / x[2]
        x = np.linspace(x[0], x[1], num=num).tolist()  # type: ignore
    if isinstance(x, tuple) and len(x) == 2:
        return x[0] + (x[1] - x[0]) * random.random()
    if isinstance(x, list):
        return random.choice(x)
    if isinstance(x, Iterable):
        raise ValueError(f"Unsupported iterable {x}")
    return x


def int_in(x: IntRange):
    if isinstance(x, tuple) and len(x) == 2:
        return random.randint(x[0], x[-1])
    if isinstance(x, tuple) and len(x) == 3:
        x = list(range(x[0], x[1], x[2]))
    if isinstance(x, list):
        return random.choice(list(x))
    if isinstance(x, Iterable):
        raise ValueError(f"Unsupported iterable {x}")
    return x


def make_trade_scenario(
    index: int,
    max_quantity: IntRange = (10, 15),
    prices: FloatIssueRange = (100, 200, 6),
    seller_target: IntRange = (2, 10),
    seller_shortfall_penalty: FloatRange = (0.1, 0.8),
    seller_excess_penalty: FloatRange = (0.01, 0.2),
    seller_quantity_weight: FloatRange = (0.1, 0.3),
    seller_price_exponent: FloatRange = [0.05, 0.2, 1.0, 2.0, 5.0],
    seller_reserved_range: FloatRange = (0.0, 0.1),
    buyer_target: IntRange = (4, 8),
    buyer_shortfall_penalty: FloatRange = (0.1, 0.8),
    buyer_excess_penalty: FloatRange = (0.1, 0.8),
    buyer_quantity_weight: FloatRange = (0.1, 0.3),
    buyer_price_exponent: FloatRange = [0.05, 0.2, 1.0, 2.0, 5.0],
    buyer_reserved_range: FloatRange = (0.0, 0.1),
    seller_starts: bool | None = True,
) -> Scenario:
    max_quantity = int_in(max_quantity)
    os = make_os(
        [
            make_issue((1, max_quantity), name="Quantity"),
            make_issue(range_in(prices), name="Price"),
        ],
        name="Trade",
    )
    assert isinstance(os, DiscreteCartesianOutcomeSpace)
    seller_starts = seller_starts or (seller_starts is None and random.random() < 0.5)
    seller_target = int_in(seller_target)
    buyer_target = int_in(buyer_target)

    seller_quantity_weight = float_in(seller_quantity_weight)
    buyer_quantity_weight = float_in(buyer_quantity_weight)

    def make_ufun(
        is_seller: bool,
        target: int,
        shortfall_penalty: FloatRange,
        excess_penalty: FloatRange,
        w: float,
        price_exponent: FloatRange,
        reserved_range: FloatRange,
        os=os,
    ) -> UtilityFunction:
        target = max(target, os.issues[0].min_value)
        target = min(target, os.issues[0].max_value)
        e = float_in(price_exponent)
        quantities = list(os.issues[0].all)
        prices = list(os.issues[1].all)

        def qval() -> list[float]:
            mn = os.issues[0].min_value
            mx = os.issues[0].max_value
            values = [0.0] * len(quantities)
            values[target - mn] = 1.0
            for q in range(target - 1, mn - 1, -1):
                penalty = float_in(shortfall_penalty)
                values[q - mn] = round(max(0.0, values[q - mn + 1] - penalty), 2)
                if values[q - mn] <= 0:
                    break
            for q in range(target + 1, mx + 1):
                penalty = float_in(excess_penalty)
                values[q - mn] = round(max(0.0, values[q - mn - 1] - penalty), 2)
                if values[q - mn] <= 0:
                    break
            return values

        def pval(is_seller) -> list[float]:
            mn, mx = os.issues[1].min_value, os.issues[1].max_value
            values = (
                [1.0 - (_ - mn) / (mx - mn) for _ in prices]
                if is_seller
                else [(_ - mn) / (mx - mn) for _ in prices]
            )
            # self.max_aspiration * (1.0 - math.pow(t, self.exponent))

            return [round(1 - math.pow(_, e), 2) for _ in values]

        return LinearAdditiveUtilityFunction(
            (
                TableFun(dict(zip(quantities, qval()))),
                TableFun(dict(zip(prices, pval(is_seller)))),
            ),
            weights=[w, 1 - w],
            reserved_value=float_in(reserved_range),
            name="Seller" if is_seller else "Buyer",
            id="Seller" if is_seller else "Buyer",
            outcome_space=os,
        )

    ufuns = [
        make_ufun(
            is_seller=True,
            target=seller_target,
            shortfall_penalty=seller_shortfall_penalty,
            excess_penalty=seller_excess_penalty,
            w=seller_quantity_weight,
            price_exponent=seller_price_exponent,
            reserved_range=seller_reserved_range,
        ),
        make_ufun(
            is_seller=False,
            target=buyer_target,
            shortfall_penalty=buyer_shortfall_penalty,
            excess_penalty=buyer_excess_penalty,
            w=buyer_quantity_weight,
            price_exponent=buyer_price_exponent,
            reserved_range=buyer_reserved_range,
        ),
    ]
    if not seller_starts:
        ufuns.reverse()

    info = load(SAMPLE_SCENRIOS / "Trade" / INFO_FILE_NAME)
    info["hints"]["Seller"]["Target Quantity"] = int(seller_target)
    info["hints"]["Buyer"]["Target Quantity"] = int(buyer_target)
    info["hints"]["Seller"]["Quantity Importance"] = round(seller_quantity_weight, 3)
    info["hints"]["Buyer"]["Quantity Importance"] = round(buyer_quantity_weight, 3)
    info["hints"]["Seller"]["Price Importance"] = round(1.0 - seller_quantity_weight, 3)
    info["hints"]["Buyer"]["Price Importance"] = round(1.0 - buyer_quantity_weight, 3)
    return Scenario(
        outcome_space=os,
        ufuns=tuple(ufuns),
        info=info,
    )


class TradeOutcomeDisplay(DefaultOutcomeDisplay):
    def str(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> str:
        if outcome is None:
            return super().str(outcome, scenario, is_done, from_human)
        return f"{outcome[0]} items at {outcome[1]}$"
