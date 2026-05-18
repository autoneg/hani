"""
Event tracking decorators and utilities for HANI.

This module provides convenient decorators and utilities to integrate
event tracking throughout the HANI application.
"""

import functools
import time
from typing import Callable, Any, Optional
import panel as pn

from hani.events import get_event_logger, EventType, log_event


def get_current_session_id() -> Optional[str]:
    """
    Get the current session ID from Panel state.

    Returns:
        Session ID if available, None otherwise
    """
    try:
        # Check if session_id is in pn.state.cache
        if hasattr(pn.state, "cache") and "session_id" in pn.state.cache:
            return pn.state.cache["session_id"]

        # Fallback: check cookies
        if hasattr(pn.state, "cookies") and "hani_session_id" in pn.state.cookies:
            return pn.state.cookies["hani_session_id"]

        return None
    except Exception:
        return None


def set_current_session_id(session_id: str):
    """
    Store the current session ID in Panel state.

    Args:
        session_id: The session identifier to store
    """
    try:
        # Store in cache (per-session)
        if hasattr(pn.state, "cache"):
            pn.state.cache["session_id"] = session_id

        # Also store in cookies for persistence
        if hasattr(pn.state, "cookies"):
            pn.state.cookies["hani_session_id"] = session_id
    except Exception as e:
        print(f"Error storing session ID: {e}")


def track_event(
    event_type: EventType | str,
    component: Optional[str] = None,
    auto_component: bool = True,
    **extra_kwargs,
):
    """
    Decorator to automatically track function calls as events.

    Args:
        event_type: Type of event to log
        component: Component name (or auto-detect from function)
        auto_component: Auto-detect component from function name
        **extra_kwargs: Additional event parameters

    Example:
        @track_event(EventType.BUTTON_CLICKED, component="LoadButton")
        def on_load_clicked(event):
            # ... load scenario ...
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            session_id = get_current_session_id()

            if session_id:
                start_time = time.time()

                # Auto-detect component name if requested
                comp = component
                if auto_component and not comp:
                    comp = func.__name__

                try:
                    # Execute the function
                    result = func(*args, **kwargs)

                    # Calculate duration
                    duration_ms = (time.time() - start_time) * 1000

                    # Log the event
                    log_event(
                        session_id=session_id,
                        event_type=event_type,
                        component=comp,
                        action="executed",
                        duration_ms=duration_ms,
                        **extra_kwargs,
                    )

                    return result

                except Exception as e:
                    # Log error event
                    duration_ms = (time.time() - start_time) * 1000
                    log_event(
                        session_id=session_id,
                        event_type=EventType.ERROR,
                        component=comp,
                        action="error",
                        value=str(e),
                        duration_ms=duration_ms,
                    )
                    raise
            else:
                # No session tracking, just execute function
                return func(*args, **kwargs)

        return wrapper

    return decorator


def track_button_click(button_name: str, **extra_kwargs):
    """
    Decorator specifically for button click handlers.

    Args:
        button_name: Name of the button
        **extra_kwargs: Additional event parameters

    Example:
        @track_button_click("Accept")
        def on_accept(event):
            # ... accept offer ...
            pass
    """
    return track_event(
        EventType.BUTTON_CLICKED,
        component=f"{button_name}Button",
        action="click",
        **extra_kwargs,
    )


def track_tool_interaction(tool_name: str, action: str = "interact", **extra_kwargs):
    """
    Decorator for tool interactions.

    Args:
        tool_name: Name of the tool
        action: Type of interaction
        **extra_kwargs: Additional event parameters

    Example:
        @track_tool_interaction("Preferences", action="view")
        def show_preferences():
            # ... show preferences ...
            pass
    """
    return track_event(
        EventType.TOOL_INTERACTION,
        component=f"{tool_name}Tool",
        action=action,
        **extra_kwargs,
    )


def log_negotiation_event(
    event_type: EventType,
    session_id: Optional[str] = None,
    offer: Optional[dict] = None,
    utility_value: Optional[float] = None,
    round_number: Optional[int] = None,
    scenario_id: Optional[str] = None,
    **kwargs,
):
    """
    Log a negotiation-specific event.

    Args:
        event_type: Type of negotiation event
        session_id: Session ID (auto-detected if None)
        offer: The offer being made/accepted/rejected
        utility_value: Utility value of the offer
        round_number: Current negotiation round
        scenario_id: Scenario identifier
        **kwargs: Additional event parameters
    """
    if session_id is None:
        session_id = get_current_session_id()

    if session_id:
        # Convert offer to JSON string if provided
        import json

        value = json.dumps(offer) if offer else None

        log_event(
            session_id=session_id,
            event_type=event_type,
            component="Negotiation",
            value=value,
            utility_value=utility_value,
            round_number=round_number,
            scenario_id=scenario_id,
            **kwargs,
        )


def log_scenario_event(
    event_type: EventType,
    scenario_id: str,
    scenario_data: Optional[dict] = None,
    session_id: Optional[str] = None,
    **kwargs,
):
    """
    Log a scenario-related event.

    Args:
        event_type: Type of scenario event
        scenario_id: Scenario identifier
        scenario_data: Scenario metadata
        session_id: Session ID (auto-detected if None)
        **kwargs: Additional event parameters
    """
    if session_id is None:
        session_id = get_current_session_id()

    if session_id:
        import json

        value = json.dumps(scenario_data) if scenario_data else None

        log_event(
            session_id=session_id,
            event_type=event_type,
            component="ScenarioManager",
            scenario_id=scenario_id,
            value=value,
            **kwargs,
        )


def create_tracked_button(name: str, **button_kwargs):
    """
    Create a Panel button with automatic click tracking.

    Args:
        name: Button name
        **button_kwargs: Additional button parameters

    Returns:
        Panel Button widget with click tracking

    Example:
        accept_button = create_tracked_button(
            "Accept",
            button_type="success",
            icon="check"
        )
        accept_button.on_click(lambda e: handle_accept(e))
    """
    button = pn.widgets.Button(name=name, **button_kwargs)

    # Wrap the on_click to add tracking
    original_on_click = button.on_click

    def tracked_on_click(callback):
        def tracked_callback(*args, **kwargs):
            session_id = get_current_session_id()
            if session_id:
                start_time = time.time()

                try:
                    result = callback(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000

                    log_event(
                        session_id=session_id,
                        event_type=EventType.BUTTON_CLICKED,
                        component=f"{name}Button",
                        action="click",
                        duration_ms=duration_ms,
                    )

                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    log_event(
                        session_id=session_id,
                        event_type=EventType.ERROR,
                        component=f"{name}Button",
                        action="click_error",
                        value=str(e),
                        duration_ms=duration_ms,
                    )
                    raise
            else:
                return callback(*args, **kwargs)

        return original_on_click(tracked_callback)

    button.on_click = tracked_on_click
    return button


def log_page_view(page_name: str, session_id: Optional[str] = None, **kwargs):
    """
    Log a page view event.

    Args:
        page_name: Name of the page being viewed
        session_id: Session ID (auto-detected if None)
        **kwargs: Additional event parameters
    """
    if session_id is None:
        session_id = get_current_session_id()

    if session_id:
        log_event(
            session_id=session_id,
            event_type=EventType.PAGE_VIEW,
            component=page_name,
            action="view",
            **kwargs,
        )


def log_modal_event(
    modal_name: str, action: str = "open", session_id: Optional[str] = None, **kwargs
):
    """
    Log modal open/close events.

    Args:
        modal_name: Name of the modal
        action: "open" or "close"
        session_id: Session ID (auto-detected if None)
        **kwargs: Additional event parameters
    """
    if session_id is None:
        session_id = get_current_session_id()

    if session_id:
        event_type = (
            EventType.MODAL_OPENED if action == "open" else EventType.MODAL_CLOSED
        )
        log_event(
            session_id=session_id,
            event_type=event_type,
            component=f"{modal_name}Modal",
            action=action,
            **kwargs,
        )
