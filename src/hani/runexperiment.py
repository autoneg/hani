"""
CLI tool for managing experiments.

Usage:
    hani-experiment create <name> [--description TEXT] [--start TIME]
    hani-experiment list [--active-only]
    hani-experiment end <experiment_id>
    hani-experiment info <experiment_id>
"""

import sys
import argparse
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from hani.events import get_event_logger


console = Console()


def create_experiment(
    name: str, description: Optional[str] = None, start_time: Optional[str] = None
) -> None:
    """Create a new experiment."""
    logger = get_event_logger()

    # Parse start time if provided
    start = datetime.now()
    if start_time:
        try:
            start = datetime.fromisoformat(start_time)
        except ValueError:
            console.print(
                f"[red]Error: Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)[/red]"
            )
            sys.exit(1)

    # Create experiment
    exp_id = logger.create_experiment(
        name=name, description=description or "", start_time=start
    )

    console.print(f"[green]Created experiment:[/green] {name}")
    console.print(f"[green]Experiment ID:[/green] {exp_id}")
    console.print(f"[green]Start time:[/green] {start.isoformat()}")


def list_experiments(active_only: bool = False) -> None:
    """List all experiments."""
    logger = get_event_logger()

    if active_only:
        experiments = logger.get_active_experiments()
        title = "Active Experiments"
    else:
        # Get all experiments (we need to query directly)
        with logger._session() as db:
            from hani.events import Experiment
            from sqlalchemy import select

            stmt = select(Experiment).order_by(Experiment.start_time.desc())
            experiments = [
                {
                    "id": exp.id,
                    "name": exp.name,
                    "description": exp.description,
                    "start_time": exp.start_time.isoformat(),
                    "end_time": exp.end_time.isoformat() if exp.end_time else None,
                    "is_active": exp.is_active,
                }
                for exp in db.execute(stmt).scalars().all()
            ]
        title = "All Experiments"

    if not experiments:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    # Create table
    table = Table(title=title)
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description", style="white")
    table.add_column("Start Time", style="yellow")
    table.add_column("End Time", style="yellow")
    table.add_column("Active", style="magenta")

    for exp in experiments:
        table.add_row(
            exp["id"][:8] + "...",  # Truncate ID for display
            exp["name"],
            exp["description"][:50] if exp["description"] else "",
            exp["start_time"][:19]
            if isinstance(exp["start_time"], str)
            else exp["start_time"].isoformat()[:19],
            exp["end_time"][:19] if exp["end_time"] else "-",
            "✓" if exp["is_active"] else "✗",
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(experiments)} experiments[/dim]")


def end_experiment(experiment_id: str) -> None:
    """End an experiment."""
    logger = get_event_logger()

    # Get experiment info first
    with logger._session() as db:
        from hani.events import Experiment
        from sqlalchemy import select

        stmt = select(Experiment).where(Experiment.id == experiment_id)
        exp = db.execute(stmt).scalar_one_or_none()

        if not exp:
            console.print(f"[red]Error: Experiment '{experiment_id}' not found.[/red]")
            sys.exit(1)

        if not exp.is_active:
            console.print(
                f"[yellow]Warning: Experiment '{exp.name}' is already ended.[/yellow]"
            )
            sys.exit(0)

    # End the experiment
    logger.end_experiment(experiment_id)
    console.print(f"[green]Ended experiment:[/green] {exp.name}")
    console.print(f"[green]Experiment ID:[/green] {experiment_id}")


def experiment_info(experiment_id: str) -> None:
    """Show detailed information about an experiment."""
    logger = get_event_logger()

    with logger._session() as db:
        from hani.events import Experiment, Session
        from sqlalchemy import select, func

        # Get experiment
        stmt = select(Experiment).where(Experiment.id == experiment_id)
        exp = db.execute(stmt).scalar_one_or_none()

        if not exp:
            console.print(f"[red]Error: Experiment '{experiment_id}' not found.[/red]")
            sys.exit(1)

        # Get session count
        session_count = db.execute(
            select(func.count(Session.id)).where(Session.experiment_id == experiment_id)
        ).scalar()

        # Get unique user count
        user_count = db.execute(
            select(func.count(func.distinct(Session.user_id))).where(
                Session.experiment_id == experiment_id
            )
        ).scalar()

    # Display info
    console.print("\n[bold cyan]Experiment Details[/bold cyan]")
    console.print(f"[green]ID:[/green] {exp.id}")
    console.print(f"[green]Name:[/green] {exp.name}")
    console.print(f"[green]Description:[/green] {exp.description or '(none)'}")
    console.print(f"[green]Start Time:[/green] {exp.start_time.isoformat()}")
    console.print(
        f"[green]End Time:[/green] {exp.end_time.isoformat() if exp.end_time else '(ongoing)'}"
    )
    console.print(f"[green]Active:[/green] {'Yes' if exp.is_active else 'No'}")
    console.print(f"[green]Sessions:[/green] {session_count}")
    console.print(f"[green]Unique Users:[/green] {user_count}")
    console.print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage HANI experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new experiment")
    create_parser.add_argument("name", help="Experiment name")
    create_parser.add_argument("--description", help="Experiment description")
    create_parser.add_argument("--start", help="Start time (ISO format, default: now)")

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument(
        "--active-only", action="store_true", help="Show only active experiments"
    )

    # End command
    end_parser = subparsers.add_parser("end", help="End an experiment")
    end_parser.add_argument("experiment_id", help="Experiment ID")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show experiment details")
    info_parser.add_argument("experiment_id", help="Experiment ID")

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        if args.command == "create":
            create_experiment(args.name, args.description, args.start)
        elif args.command == "list":
            list_experiments(args.active_only)
        elif args.command == "end":
            end_experiment(args.experiment_id)
        elif args.command == "info":
            experiment_info(args.experiment_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
