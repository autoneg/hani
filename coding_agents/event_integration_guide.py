"""
Integration patches for adding event tracking to HANI app.py

This module provides the necessary modifications to integrate
comprehensive event tracking into the main HANI application.

Apply these changes to src/hani/app.py to enable full event tracking.
"""

# INTEGRATION INSTRUCTIONS:
# ========================
#
# 1. Add imports at the top of app.py (after existing imports):
#
# from hani.events import EventType, create_session, end_session
# from hani.event_tracking import (
#     get_current_session_id,
#     set_current_session_id,
#     log_negotiation_event,
#     log_scenario_event,
#     log_page_view,
#     create_tracked_button,
#     track_button_click,
# )
#
# 2. Initialize session on app load (in main() or template function):
#
# @pn.cache(per_session=True)
# def init_session():
#     """Initialize event tracking session for current user."""
#     if pn.state.user:
#         # Get user info
#         user_id = pn.state.user
#         ip_address = None
#         user_agent = None
#
#         # Try to get request info
#         try:
#             if hasattr(pn.state, 'headers'):
#                 user_agent = pn.state.headers.get('User-Agent')
#             if hasattr(pn.state, 'cookies'):
#                 # Get IP from headers or cookies if available
#                 pass
#         except:
#             pass
#
#         # Create session
#         session_id = create_session(
#             user_id=user_id,
#             ip_address=ip_address,
#             user_agent=user_agent
#         )
#
#         # Store session ID
#         set_current_session_id(session_id)
#
#         # Log page view
#         log_page_view("MainApp", session_id=session_id)
#
#     return True
#
# 3. Add session cleanup on logout:
#
# def handle_logout():
#     """Handle logout and end session."""
#     session_id = get_current_session_id()
#     if session_id:
#         end_session(session_id)
#
# 4. Track scenario loading (in load_scenario function):
#
# def load_scenario(...):
#     # ... existing code ...
#
#     # Log scenario loaded event
#     log_scenario_event(
#         event_type=EventType.SCENARIO_LOADED,
#         scenario_id=scenario_name,
#         scenario_data={
#             "name": scenario_name,
#             "type": scenario_type,
#             # ... other metadata ...
#         }
#     )
#
#     # ... rest of function ...
#
# 5. Track negotiation start:
#
# def start_negotiation(...):
#     # ... existing code ...
#
#     # Log scenario started
#     log_scenario_event(
#         event_type=EventType.SCENARIO_STARTED,
#         scenario_id=session_state.get("scenario_id"),
#     )
#
#     # ... rest of function ...
#
# 6. Track negotiation actions (offers, accepts, rejects):
#
# def on_accept(event):
#     """Handle accept button click."""
#     session_id = get_current_session_id()
#
#     # ... existing accept logic ...
#
#     # Log acceptance event
#     log_negotiation_event(
#         event_type=EventType.OFFER_ACCEPTED,
#         session_id=session_id,
#         offer=current_offer,
#         utility_value=utility,
#         round_number=current_round,
#         scenario_id=session_state.get("scenario_id")
#     )
#
# def on_reject(event):
#     """Handle reject button click."""
#     session_id = get_current_session_id()
#
#     # ... existing reject logic ...
#
#     # Log rejection event
#     log_negotiation_event(
#         event_type=EventType.OFFER_REJECTED,
#         session_id=session_id,
#         offer=current_offer,
#         utility_value=utility,
#         round_number=current_round,
#         scenario_id=session_state.get("scenario_id")
#     )
#
# def make_counter_offer(offer):
#     """Handle counter-offer submission."""
#     session_id = get_current_session_id()
#
#     # ... existing counter-offer logic ...
#
#     # Log counter-offer event
#     log_negotiation_event(
#         event_type=EventType.COUNTER_OFFER,
#         session_id=session_id,
#         offer=offer,
#         utility_value=utility,
#         round_number=current_round,
#         scenario_id=session_state.get("scenario_id")
#     )
#
# 7. Replace regular buttons with tracked buttons:
#
# # OLD:
# accept_btn = pn.widgets.Button(name="Accept", button_type="success")
#
# # NEW:
# accept_btn = create_tracked_button(name="Accept", button_type="success")
#
# 8. Track negotiation end:
#
# def end_negotiation(outcome):
#     """Handle negotiation end."""
#     session_id = get_current_session_id()
#
#     # ... existing end logic ...
#
#     # Log negotiation end
#     log_negotiation_event(
#         event_type=EventType.NEGOTIATION_ENDED,
#         session_id=session_id,
#         offer=outcome,
#         utility_value=final_utility,
#         round_number=total_rounds,
#         scenario_id=session_state.get("scenario_id")
#     )
#
# 9. Track tool usage (in Tool base class):
#
# In src/hani/tools/tool.py:
#
# from hani.event_tracking import log_event, get_current_session_id
# from hani.events import EventType
#
# class Tool:
#     def panel(self, session_state):
#         """Create panel view with event tracking."""
#         session_id = get_current_session_id()
#
#         if session_id:
#             log_event(
#                 session_id=session_id,
#                 event_type=EventType.TOOL_OPENED,
#                 component=self.__class__.__name__,
#                 action="view"
#             )
#
#         # ... existing panel creation ...
#
# 10. Add session initialization to template:
#
# In the main app template function, add:
#
# # Initialize session tracking
# init_session()
#
# template.main.append(...)

# QUICK START EXAMPLE:
# ===================
#
# Here's a minimal working example to add to app.py:

MINIMAL_INTEGRATION = '''
# At top of file, add imports:
from hani.events import EventType, create_session, end_session
from hani.event_tracking import (
    get_current_session_id,
    set_current_session_id,
    log_negotiation_event,
    create_tracked_button,
)

# Add session init function:
@pn.cache(per_session=True)
def init_event_tracking():
    """Initialize event tracking for this session."""
    if pn.state.user:
        session_id = create_session(user_id=str(pn.state.user))
        set_current_session_id(session_id)
        print(f"Event tracking initialized: {session_id}")
    return True

# In your main template/app function, add early on:
init_event_tracking()

# Replace button creation:
# OLD: accept_btn = pn.widgets.Button(name="Accept", button_type="success")
# NEW: accept_btn = create_tracked_button(name="Accept", button_type="success")

# In accept/reject/counter-offer handlers, add:
def on_accept(event):
    log_negotiation_event(
        event_type=EventType.OFFER_ACCEPTED,
        offer={"current": "offer", "data": "here"},
        utility_value=0.75,
        round_number=5
    )
    # ... rest of logic ...
'''

print(__doc__)
print(MINIMAL_INTEGRATION)
