#!/usr/bin/env python3
from negmas.checkpoints import shutil
import typer
from typing import Annotated
from negmas.inout import Scenario
from pathlib import Path
from hani.scenarios.trade import make_trade_scenario
from hani.scenarios.island import make_island_scenario
from hani.scenarios.grocery import make_grocery_scenario

app = typer.Typer()

MAKER_MAP = {
    "Trade": make_trade_scenario,
    "Island": make_island_scenario,
    "Grocery": make_grocery_scenario,
}
BASE = Path(__file__).parent / "sample_scenarios"
SRC = Path(__file__).parent / "sample_scenarios" / "Default"
files = ["display.py"]


def main(
    scenario_type: Annotated[
        str,
        typer.Argument(
            help=f"Type to create. Available options are {list(MAKER_MAP.keys())}"
        ),
    ],
    n: Annotated[int, typer.Option("-n", help="Number of scenarios to create")] = 20,
    base: Annotated[Path, typer.Option(help="Base directory for scenarios")] = BASE,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", "-o", help="Overwrite existing scenarios")
    ] = False,
    dry: Annotated[
        bool, typer.Option("--overwrite", "-o", help="Overwrite existing scenarios")
    ] = False,
):
    last_index = 0
    type_base = base / scenario_type
    if type_base.exists():
        numbers = [int(_.name[:4]) for _ in type_base.iterdir() if _.is_dir()]
        if numbers:
            last_index = max(numbers)
            print(
                f"Found {last_index} scenarios in {type_base}. Will start from {last_index + 1}"
            )
    print(f"Will start creating scenarios in {type_base}... with {last_index + 1}")
    # for i in track(range(last_index + 1, last_index + n + 1), "Creating scenarios"):
    for i in range(last_index + 1, last_index + n + 1):
        maker = MAKER_MAP.get(scenario_type)
        if not maker:
            typer.echo(
                f"Unknown type: {scenario_type}. Available options are {list(MAKER_MAP.keys())}"
            )
            raise typer.Exit(code=1)
        scenario: Scenario = maker(i)
        info = scenario.info
        fname = f"{i:04d}{scenario_type}"
        path = type_base / fname
        if path.exists() and not overwrite:
            print(f"Scenario {path} already exists. Skipping...")
            continue
        if dry:
            print("Will create {path}")
            continue
        path.mkdir(exist_ok=True, parents=True)
        scenario.to_yaml(path)
        for f in files:
            shutil.copyfile(SRC / scenario_type / f, path / f)
        # dump(info, path / "_info.yml", f)


if __name__ == "__main__":
    typer.run(main)
