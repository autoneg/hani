"""
Event Analytics and Visualization Dashboard for HANI.

This module provides a Panel-based web application for visualizing and analyzing
event tracking data from user sessions.
"""

import panel as pn
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional

from hani.events import get_event_logger, EventType

# Configure Panel
pn.extension("plotly", "tabulator", sizing_mode="stretch_width")


class EventAnalyticsDashboard:
    """Dashboard for analyzing event tracking data."""

    def __init__(self):
        self.logger = get_event_logger()

        # Widgets
        self.experiment_selector = pn.widgets.Select(
            name="Experiment", options=["All Experiments"], value="All Experiments"
        )
        self.user_selector = pn.widgets.Select(
            name="User",
            options=["All Users"] + self.logger.get_all_users(),
            value="All Users",
        )
        self.session_selector = pn.widgets.Select(
            name="Session", options=["All Sessions"], value="All Sessions"
        )
        self.event_type_selector = pn.widgets.MultiChoice(
            name="Event Types", options=[e.value for e in EventType], value=[]
        )
        self.date_range_picker = pn.widgets.DatetimeRangePicker(
            name="Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh Data", button_type="primary", icon="refresh"
        )
        self.export_button = pn.widgets.Button(
            name="Export to CSV", button_type="success", icon="download"
        )

        # Current session info pane
        self.current_session_pane = pn.pane.Markdown("")

        # Load experiments
        self._load_experiments()

        # Bind callbacks
        self.experiment_selector.param.watch(self._on_experiment_change, "value")
        self.user_selector.param.watch(self._on_user_change, "value")
        self.refresh_button.on_click(self._on_refresh)
        self.export_button.on_click(self._on_export)

        # Data containers
        self.stats_pane = pn.pane.Markdown("Loading statistics...")
        self.event_count_plot = pn.pane.Plotly()
        self.timeline_plot = pn.pane.Plotly()
        self.event_table = pn.widgets.Tabulator(pagination="local", page_size=20)
        self.session_table = pn.widgets.Tabulator(pagination="local", page_size=10)

        # Initial load
        self._load_data()
        self._update_current_session_info()

    def _load_experiments(self):
        """Load all experiments into the selector."""
        experiments = self.logger.get_active_experiments()
        all_experiments = []

        # Get all experiments (active and inactive)
        with self.logger._session() as db:
            from hani.events import Experiment
            from sqlalchemy import select

            stmt = select(Experiment).order_by(Experiment.start_time.desc())
            all_experiments = [
                {"id": exp.id, "name": exp.name, "is_active": exp.is_active}
                for exp in db.execute(stmt).scalars().all()
            ]

        if all_experiments:
            exp_options = ["All Experiments"] + [
                f"{exp['name']}{' (Active)' if exp['is_active'] else ''}"
                for exp in all_experiments
            ]
            self.experiment_selector.options = exp_options
            # Store mapping
            self._experiment_map = {
                f"{exp['name']}{' (Active)' if exp['is_active'] else ''}": exp["id"]
                for exp in all_experiments
            }
        else:
            self.experiment_selector.options = ["All Experiments"]
            self._experiment_map = {}

    def _on_experiment_change(self, event):
        """Handle experiment selection change."""
        self._load_data()

    def _update_current_session_info(self):
        """Update the current session information panel."""
        info = "## Current Session Info\n\n"

        # Try to get current session from Panel state
        try:
            import panel as pn

            if hasattr(pn.state, "cache"):
                experiment_id = pn.state.cache.get("experiment_id")
                if experiment_id:
                    # Get experiment info
                    with self.logger._session() as db:
                        from hani.events import Experiment
                        from sqlalchemy import select

                        stmt = select(Experiment).where(Experiment.id == experiment_id)
                        exp = db.execute(stmt).scalar_one_or_none()

                        if exp:
                            info += f"**Experiment:** {exp.name}\n\n"
                        else:
                            info += f"**Experiment ID:** {experiment_id[:8]}...\n\n"

                # Get session ID from event tracking
                from hani.event_tracking import get_current_session_id

                session_id = get_current_session_id()

                if session_id:
                    info += f"**Session ID:** {session_id[:8]}...\n\n"

                    # Get session details
                    with self.logger._session() as db:
                        from hani.events import Session
                        from sqlalchemy import select

                        stmt = select(Session).where(Session.id == session_id)
                        session = db.execute(stmt).scalar_one_or_none()

                        if session:
                            info += f"**User:** {session.user_id}\n\n"
                            info += f"**Started:** {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                            if session.end_time:
                                info += f"**Ended:** {session.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                            else:
                                info += "**Status:** Active\n\n"
                else:
                    info += "*No active session*\n\n"
        except Exception as e:
            info += f"*Could not load session info: {e}*\n\n"

        self.current_session_pane.object = info

    def _on_user_change(self, event):
        """Handle user selection change."""
        user = event.new
        if user == "All Users":
            self.session_selector.options = ["All Sessions"]
            self.session_selector.value = "All Sessions"
        else:
            sessions = self.logger.get_user_sessions(user)
            session_options = ["All Sessions"] + [
                f"{s['id'][:8]}... ({s['start_time'][:19]})" for s in sessions
            ]
            self.session_selector.options = session_options
            self.session_selector.value = "All Sessions"
        self._load_data()

    def _on_refresh(self, event):
        """Refresh all data."""
        self.user_selector.options = ["All Users"] + self.logger.get_all_users()
        self._load_data()

    def _on_export(self, event):
        """Export current data to CSV."""
        df = self._get_events_dataframe()
        if df is not None and not df.empty:
            filename = f"events_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            pn.state.notifications.success(f"Exported to {filename}")
        else:
            pn.state.notifications.warning("No data to export")

    def _get_events_dataframe(self) -> Optional[pd.DataFrame]:
        """Get events as a pandas DataFrame."""
        user = (
            self.user_selector.value
            if self.user_selector.value != "All Users"
            else None
        )

        # Get experiment filter
        experiment_id = None
        if self.experiment_selector.value != "All Experiments":
            experiment_id = self._experiment_map.get(self.experiment_selector.value)

        session_id = None

        if user:
            sessions = self.logger.get_user_sessions(user, experiment_id=experiment_id)
            if sessions:
                # Get all events for all sessions of this user
                all_events = []
                for session in sessions:
                    events = self.logger.get_session_events(session["id"])
                    all_events.extend(events)

                if all_events:
                    df = pd.DataFrame(all_events)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    return df
        else:
            # Get all users and their events
            all_events = []
            for user_id in self.logger.get_all_users():
                sessions = self.logger.get_user_sessions(
                    user_id, experiment_id=experiment_id
                )
                for session in sessions:
                    events = self.logger.get_session_events(session["id"])
                    all_events.extend(events)

            if all_events:
                df = pd.DataFrame(all_events)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df

        return None

    def _load_data(self):
        """Load and display all data."""
        user = (
            self.user_selector.value
            if self.user_selector.value != "All Users"
            else None
        )

        # Get experiment filter
        experiment_id = None
        if self.experiment_selector.value != "All Experiments":
            experiment_id = self._experiment_map.get(self.experiment_selector.value)

        # Load statistics
        if user:
            stats = self.logger.get_event_stats(user_id=user)
        else:
            stats = self.logger.get_event_stats()

        self._update_stats(stats)

        # Load events data
        df = self._get_events_dataframe()
        if df is not None and not df.empty:
            self._update_event_count_plot(df)
            self._update_timeline_plot(df)
            self._update_event_table(df)
        else:
            self.stats_pane.object = (
                "## No data available\n\nNo events found for the selected filters."
            )
            self.event_count_plot.object = None
            self.timeline_plot.object = None
            self.event_table.value = pd.DataFrame()

        # Load sessions data
        if user:
            sessions = self.logger.get_user_sessions(user, experiment_id=experiment_id)
        else:
            sessions = []
            for user_id in self.logger.get_all_users():
                sessions.extend(
                    self.logger.get_user_sessions(user_id, experiment_id=experiment_id)
                )

        if sessions:
            self._update_session_table(sessions)
        else:
            self.session_table.value = pd.DataFrame()

    def _update_stats(self, stats: dict):
        """Update statistics panel."""
        md = f"""
## Event Statistics

- **Total Events**: {stats.get("total_events", 0)}
- **Average Duration**: {stats.get("avg_duration_ms", 0):.2f} ms
- **First Event**: {stats.get("first_event", "N/A")}
- **Last Event**: {stats.get("last_event", "N/A")}

### Event Counts by Type

"""
        event_counts = stats.get("event_counts", {})
        for event_type, count in sorted(
            event_counts.items(), key=lambda x: x[1], reverse=True
        ):
            md += f"- **{event_type}**: {count}\n"

        self.stats_pane.object = md

    def _update_event_count_plot(self, df: pd.DataFrame):
        """Update event count bar chart."""
        event_counts = df["event_type"].value_counts()

        fig = go.Figure(
            data=[
                go.Bar(
                    x=event_counts.index,
                    y=event_counts.values,
                    marker_color="steelblue",
                )
            ]
        )

        fig.update_layout(
            title="Event Counts by Type",
            xaxis_title="Event Type",
            yaxis_title="Count",
            height=400,
            margin=dict(l=50, r=50, t=50, b=100),
            xaxis_tickangle=-45,
        )

        self.event_count_plot.object = fig

    def _update_timeline_plot(self, df: pd.DataFrame):
        """Update timeline plot."""
        # Group by hour and event type
        df["hour"] = df["timestamp"].dt.floor("H")
        hourly_counts = (
            df.groupby(["hour", "event_type"]).size().reset_index(name="count")
        )

        fig = go.Figure()

        for event_type in hourly_counts["event_type"].unique():
            type_data = hourly_counts[hourly_counts["event_type"] == event_type]
            fig.add_trace(
                go.Scatter(
                    x=type_data["hour"],
                    y=type_data["count"],
                    mode="lines+markers",
                    name=event_type,
                )
            )

        fig.update_layout(
            title="Event Timeline (by Hour)",
            xaxis_title="Time",
            yaxis_title="Event Count",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode="x unified",
        )

        self.timeline_plot.object = fig

    def _update_event_table(self, df: pd.DataFrame):
        """Update events table."""
        # Prepare display dataframe
        display_df = df[
            [
                "timestamp",
                "event_type",
                "component",
                "action",
                "scenario_id",
                "round_number",
                "utility_value",
                "duration_ms",
            ]
        ].copy()

        # Format timestamp
        display_df["timestamp"] = display_df["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Sort by timestamp descending
        display_df = display_df.sort_values("timestamp", ascending=False)

        self.event_table.value = display_df

    def _update_session_table(self, sessions: list):
        """Update sessions table."""
        if not sessions:
            self.session_table.value = pd.DataFrame()
            return

        df = pd.DataFrame(sessions)

        # Calculate session duration
        df["duration"] = pd.to_datetime(df["end_time"]) - pd.to_datetime(
            df["start_time"]
        )
        df["duration"] = df["duration"].apply(
            lambda x: str(x).split(".")[0] if pd.notna(x) else "Active"
        )

        # Get experiment names
        if "experiment_id" in df.columns:
            # Create experiment name mapping
            exp_names = {}
            with self.logger._session() as db:
                from hani.events import Experiment
                from sqlalchemy import select

                for exp_id in df["experiment_id"].unique():
                    stmt = select(Experiment).where(Experiment.id == exp_id)
                    exp = db.execute(stmt).scalar_one_or_none()
                    if exp:
                        exp_names[exp_id] = exp.name
                    else:
                        exp_names[exp_id] = exp_id[:8] + "..."

            df["experiment_name"] = df["experiment_id"].map(exp_names)

            # Select and rename columns
            display_df = df[
                [
                    "user_id",
                    "experiment_name",
                    "start_time",
                    "end_time",
                    "duration",
                    "is_active",
                ]
            ].copy()

            display_df.columns = [
                "User",
                "Experiment",
                "Start Time",
                "End Time",
                "Duration",
                "Active",
            ]
        else:
            # Select and rename columns (old format without experiment)
            display_df = df[
                ["user_id", "start_time", "end_time", "duration", "is_active"]
            ].copy()

            display_df.columns = [
                "User",
                "Start Time",
                "End Time",
                "Duration",
                "Active",
            ]

        self.session_table.value = display_df

    def create_layout(self):
        """Create the dashboard layout."""
        # Sidebar with filters
        sidebar = pn.Column(
            "# Event Analytics",
            "---",
            self.current_session_pane,
            "---",
            "## Filters",
            self.experiment_selector,
            self.user_selector,
            self.session_selector,
            self.event_type_selector,
            self.date_range_picker,
            pn.Row(self.refresh_button, self.export_button),
            "---",
            self.stats_pane,
            width=350,
            scroll=True,
        )

        # Main content area
        main_content = pn.Tabs(
            ("Overview", pn.Column(self.event_count_plot, self.timeline_plot)),
            ("Events", pn.Column("## Event Details", self.event_table)),
            ("Sessions", pn.Column("## Session Details", self.session_table)),
        )

        # Full layout
        template = pn.template.FastListTemplate(
            title="HANI Event Analytics Dashboard",
            sidebar=[sidebar],
            main=[main_content],
            accent_base_color="#0072B5",
            header_background="#0072B5",
        )

        return template


def create_dashboard():
    """Create and return the dashboard."""
    dashboard = EventAnalyticsDashboard()
    return dashboard.create_layout()


# For running with panel serve
if __name__ == "__main__":
    create_dashboard().servable()
elif __name__.startswith("bokeh"):
    create_dashboard().servable()
