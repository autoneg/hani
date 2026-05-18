"""
Experiment selector UI component for HANI.

This module provides a Panel interface for users to select which experiment
they want to participate in after logging in.
"""

import panel as pn
from typing import Optional
from hani.events import get_event_logger


def create_experiment_selector() -> Optional[str]:
    """
    Create and display an experiment selector interface.

    Returns:
        The selected experiment ID, or None if no selection was made.
    """
    logger = get_event_logger()

    # Get active experiments
    experiments = logger.get_active_experiments()

    if not experiments:
        # No active experiments - create a default one
        print("No active experiments found. Creating default experiment...")
        from datetime import datetime

        exp_id = logger.create_experiment(
            name="Default Experiment",
            description="Auto-created default experiment",
            start_time=datetime.now(),
        )
        if hasattr(pn.state, "notifications") and pn.state.notifications:
            pn.state.notifications.success(
                "Created and selected default experiment", duration=3000
            )
        return exp_id

    # If only one experiment, auto-select it
    if len(experiments) == 1:
        exp = experiments[0]
        if hasattr(pn.state, "notifications") and pn.state.notifications:
            pn.state.notifications.success(
                f"Automatically selected experiment: {exp['name']}", duration=3000
            )
        return exp["id"]

    # Multiple experiments - let user choose
    selected_id = None

    # Create radio button group with experiment names
    experiment_options = {
        f"{exp['name']} - {exp['description'][:100]}": exp["id"] for exp in experiments
    }

    selector = pn.widgets.RadioButtonGroup(
        name="Select Experiment",
        options=list(experiment_options.keys()),
        button_type="primary",
    )

    description_pane = pn.pane.Markdown(
        "## Select an Experiment\n\nPlease choose which experiment you want to participate in:"
    )

    submit_button = pn.widgets.Button(name="Continue", button_type="success", width=200)

    status_pane = pn.pane.Markdown("")

    def on_submit(event):
        """Handle experiment selection."""
        nonlocal selected_id

        if not selector.value:
            status_pane.object = "⚠️ **Please select an experiment before continuing.**"
            return

        # Get the selected experiment ID
        selected_id = experiment_options[selector.value]

        # Store in session state
        pn.state.cache["experiment_id"] = selected_id

        # Find experiment name for confirmation
        exp_name = next(e["name"] for e in experiments if e["id"] == selected_id)

        if hasattr(pn.state, "notifications") and pn.state.notifications:
            pn.state.notifications.success(
                f"Selected experiment: {exp_name}", duration=2000
            )

        # Signal that selection is complete
        pn.state.cache["experiment_selected"] = True

    submit_button.on_click(on_submit)

    # Create layout
    layout = pn.Column(
        description_pane,
        pn.layout.Divider(),
        selector,
        pn.layout.Divider(),
        submit_button,
        status_pane,
        width=800,
        styles={
            "background": "white",
            "padding": "20px",
            "border-radius": "10px",
            "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
    )

    # Wait for user to make selection
    # This is a blocking display - the function won't return until selection is made
    # Note: In Panel, we'll need to handle this differently in the actual app flow

    return layout


def get_selected_experiment_id() -> Optional[str]:
    """
    Get the currently selected experiment ID from session state.

    Returns:
        The experiment ID, or None if not set.
    """
    return pn.state.cache.get("experiment_id")


def ensure_experiment_selected(user_id: str) -> str:
    """
    Ensure that an experiment has been selected for the current user.

    This function checks if an experiment has been selected. If not, it prompts
    the user to select one. If only one active experiment exists, it auto-selects it.
    If no experiments exist, creates a default one.

    Args:
        user_id: The username of the current user

    Returns:
        The selected experiment ID
    """
    logger = get_event_logger()

    # Check if already selected in this session
    if "experiment_id" in pn.state.cache:
        return pn.state.cache["experiment_id"]

    # Get active experiments
    experiments = logger.get_active_experiments()

    # If no experiments exist, create a default one
    if not experiments:
        print("No active experiments found. Creating default experiment...")
        from datetime import datetime

        exp_id = logger.create_experiment(
            name="Default Experiment",
            description="Auto-created default experiment",
            start_time=datetime.now(),
        )
        pn.state.cache["experiment_id"] = exp_id
        print(f"Created and selected default experiment for user {user_id}")
        return exp_id

    # If only one experiment, auto-select it
    if len(experiments) == 1:
        exp_id = experiments[0]["id"]
        pn.state.cache["experiment_id"] = exp_id
        print(f"Auto-selected experiment: {experiments[0]['name']} for user {user_id}")
        return exp_id

    # Multiple experiments - need user to select
    # This should be handled by the UI before calling this function
    raise RuntimeError(
        "Multiple experiments available - user must select one. "
        "This should have been handled by the experiment selector UI."
    )


def show_experiment_selector_modal():
    """
    Show the experiment selector as a modal dialog.

    This is designed to be called after login but before the main app loads.
    """
    selector_layout = create_experiment_selector()

    # Create modal
    modal = pn.Column(
        "# Welcome to HANI",
        selector_layout,
        styles={"background": "#f5f5f5", "padding": "40px", "min-height": "100vh"},
    )

    return modal
