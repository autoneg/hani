#!/usr/bin/env python3
import pandas as pd
from collections import defaultdict
from rich.progress import Progress, track
from negmas import LinearUtilityAggregationFunction
from negmas.helpers.inout import dump
from rich import print
from negmas.inout import Scenario
from pathlib import Path
import typer
from typing import Annotated

app = typer.Typer()

SCENARIOS_BASE = (
    Path.home()
    / "negmas"
    / "hani"
    / "ijcai2025"
    / "ijcai_scenarios"
    / "sample_scenarios"
)
RESULTS_BASE = Path.home() / "negmas" / "hani" / "ijcai2025" / "db"
DST = Path.home() / "negmas" / "hani" / "ijcai2025" / "id_scenario.yaml"
INV = Path.home() / "negmas" / "hani" / "ijcai2025" / "scenario_ids.yaml"
DST_UNMATCHED = (
    Path.home() / "negmas" / "hani" / "ijcai2025" / "id_scenario_unmatched.yaml"
)
RESULTS_FILE = "results.csv"


def is_equal(s1: Scenario, s2: Scenario, eps: float = 1e-6) -> bool:
    if s1.outcome_space.name != s2.outcome_space.name:
        return False
    name = s1.outcome_space.name
    if len(s1.outcome_space.issues) != len(s2.outcome_space.issues):
        return False
    if len(s1.ufuns) != len(s2.ufuns):
        return False
    if not (
        s1.outcome_space in s2.outcome_space and s2.outcome_space in s1.outcome_space
    ):
        return False
    for u1, u2 in zip(s1.ufuns, s2.ufuns):
        assert isinstance(u1, LinearUtilityAggregationFunction) and isinstance(
            u2, LinearUtilityAggregationFunction
        )
        # Check if the utility functions are equivalent by sampling some points
        if any((w1 - w2) ** 2 > eps for w1, w2 in zip(u1.weights, u2.weights)):
            return False
        if name in ("Island", "Grocery"):
            return True
        for v1, v2 in zip(u1.values, u2.values):
            m1, m2 = v1.mapping, v2.mapping
            if len(m1) != len(m2):
                return False
            for k, mm1 in m1.items():
                if k not in m2:
                    return False
                if abs(mm1 - m2[k]) > eps:
                    return False

    return True


def main(
    scenarios: Annotated[
        Path, typer.Option(help="Base directory for scenarios storage")
    ] = SCENARIOS_BASE,
    results: Annotated[
        Path, typer.Option(help="Base directory for results storage")
    ] = RESULTS_BASE,
    dst: Annotated[
        Path, typer.Option(help="Where to save the map from ID to specific scenario")
    ] = DST,
    inv: Annotated[
        Path,
        typer.Option(help="Where to save the inverse map specific scenario to IDs"),
    ] = INV,
    unmatched: Annotated[
        Path, typer.Option(help="Where to save the map")
    ] = DST_UNMATCHED,
    update_results: Annotated[
        bool, typer.Option(help="Whether to update results")
    ] = False,
    verbose: bool = True,
):
    type_folders = [_ for _ in scenarios.iterdir() if _.is_dir()]

    base_folders = []
    for type_folder in type_folders:
        base_folders.extend([_ for _ in type_folder.iterdir() if _.is_dir()])

    base_scenarios = [Scenario.load(_) for _ in base_folders]
    base_names = set(_.name for _ in base_folders)

    name_index = defaultdict(list)
    type_folder = defaultdict(list)
    type_scenario = defaultdict(list)
    for i, (s, b) in enumerate(zip(base_scenarios, base_folders)):
        assert s is not None, f"Failed to load scenario from {b}"
        name_index[b.name].append(i)
        type_folder[s.outcome_space.name].append(b)
        type_scenario[s.outcome_space.name].append(s)

    user_folders = [_ for _ in RESULTS_BASE.iterdir() if _.is_dir()]
    final_map = dict()
    unmatched_map = dict()
    inv_map = defaultdict(list)
    nxt = dict(Grocery=1000, Trade=1000, Island=1000)

    with Progress() as progress:
        user_task = progress.add_task("[cyan]User folders...", total=len(user_folders))
        for user_folder in user_folders:
            sfolder = user_folder / "scenarios"
            if not sfolder.is_dir():
                if verbose:
                    print(f"No scenarios folder found in {user_folder}")
                progress.advance(user_task)
                continue
            user_scenarios = {
                _.name: Scenario.load(_) for _ in sfolder.iterdir() if _.is_dir()
            }
            scenario_task = progress.add_task(
                f"[green]Scenarios in {user_folder.name}...", total=len(user_scenarios)
            )
            for id, scenario in user_scenarios.items():
                base_task = progress.add_task(
                    f"[magenta]Comparing {id}...", total=len(base_scenarios)
                )
                base_scenario = None
                for i, base_scenario in enumerate(base_scenarios):
                    assert scenario is not None, f"Failed to load scenario from {id}"
                    assert base_scenario is not None, (
                        f"Failed to load scenario from {base_folders[i]}"
                    )
                    if is_equal(scenario, base_scenario):
                        final_map[id] = base_folders[i].name
                        inv_map[base_folders[i].name].append(id)
                        progress.advance(base_task)
                        break
                    progress.advance(base_task)
                else:
                    if verbose:
                        print(f"No mach found for {id}")
                    if base_scenario is not None:
                        assert (
                            base_scenario.outcome_space is not None
                            and base_scenario.outcome_space.name is not None
                        )
                        unmatched_map[id] = (
                            f"{nxt[base_scenario.outcome_space.name]:04d}{base_scenario.outcome_space.name}"
                        )
                        nxt[base_scenario.outcome_space.name] += 1

                progress.remove_task(base_task)
                progress.advance(scenario_task)
            progress.remove_task(scenario_task)
            progress.advance(user_task)
        progress.remove_task(user_task)

    dump(final_map, dst)
    dump(unmatched_map, unmatched)
    dump(inv_map, inv)
    print(f"found {len(final_map)} matched scenarios and saved to {dst}")
    print(f"found {len(unmatched_map)} unmatched scenarios and saved to {unmatched}")
    if update_results:
        files = list(RESULTS_BASE.glob("**/results.csv"))
        for f in track(files, "Updating results"):
            data = pd.read_csv(f, index_col=False)
            data["mechanism_id"] = data["id"]
            data["scenario_name"] = data["id"].apply(
                lambda x: final_map.get(x, unmatched_map.get(x, "Unknown"))
            )
            data["load_index"] = data["scenario_name"].apply(
                lambda x: int(x[:4]) if x != "Unknown" else -1
            )
            data["load_path"] = data["scenario_name"].apply(
                lambda x: f"/Users/yasser/negmas/hani/settings/scenarios/{x[4:]}/{x}"
            )
            data["generated"] = data["scenario_name"].apply(
                lambda x: x not in base_names
            )
            data.to_csv(f, index=False)
    # results_files = results.glob("**/results.csv")
    # for f in results_files:
    #     data = pd.read_csv(f, index_col=False)


if __name__ == "__main__":
    typer.run(main)
