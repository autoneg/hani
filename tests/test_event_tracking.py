"""
Tests for the event tracking system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

from hani.events import (
    EventLogger,
    EventType,
    Session,
    Event,
    Experiment,
    User,
    create_session,
    end_session,
    log_event,
    get_event_logger,
)


@pytest.fixture
def temp_db_path(monkeypatch):
    """Create a temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    # Monkeypatch DB_PATH in hani.common
    monkeypatch.setattr("hani.common.DB_PATH", temp_path)
    monkeypatch.setattr("hani.events.DB_PATH", temp_path)

    yield temp_path

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def event_logger(temp_db_path, monkeypatch):
    """Create a fresh event logger for testing."""
    import hani.events as events_module

    # Monkeypatch DATABASE_URL to use temp path
    temp_db_url = f"sqlite:///{temp_db_path / 'events.db'}"
    monkeypatch.setattr("hani.events.DATABASE_URL", temp_db_url)

    # Reset the singleton
    EventLogger._instance = None
    EventLogger._initialized = False
    events_module._event_logger = None

    # Create logger which will use temp_db_path
    logger = EventLogger()

    # CRITICAL: Update global _event_logger so convenience functions work
    events_module._event_logger = logger

    yield logger

    # Cleanup: close connections
    if hasattr(logger, "engine"):
        logger.engine.dispose()

    # Reset singleton after test
    EventLogger._instance = None
    EventLogger._initialized = False
    events_module._event_logger = None


@pytest.fixture
def experiment_id(event_logger):
    """Create a default experiment for testing."""
    return event_logger.create_experiment(
        name="Test Experiment",
        description="Default test experiment",
        start_time=datetime.now(),
    )


class TestEventLogger:
    """Test the EventLogger class."""

    def test_singleton_pattern(self, event_logger):
        """Test that EventLogger follows singleton pattern."""
        logger1 = get_event_logger()
        logger2 = get_event_logger()
        assert logger1 is logger2

    def test_create_session(self, event_logger, experiment_id):
        """Test creating a session."""
        session_id = event_logger.create_session(
            user_id="test_user",
            experiment_id=experiment_id,
            ip_address="127.0.0.1",
            user_agent="Test Browser",
        )

        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID length

        # Verify session was created in database
        sessions = event_logger.get_user_sessions("test_user")
        assert len(sessions) == 1
        assert sessions[0]["id"] == session_id
        assert sessions[0]["user_id"] == "test_user"
        assert sessions[0]["experiment_id"] == experiment_id
        assert sessions[0]["is_active"] is True

    def test_end_session(self, event_logger, experiment_id):
        """Test ending a session."""
        session_id = event_logger.create_session(
            user_id="test_user", experiment_id=experiment_id
        )

        # End the session
        event_logger.end_session(session_id)

        # Verify session was ended
        sessions = event_logger.get_user_sessions("test_user")
        assert len(sessions) == 1
        assert sessions[0]["is_active"] is False
        assert sessions[0]["end_time"] is not None

    def test_log_event(self, event_logger, experiment_id):
        """Test logging an event."""
        session_id = event_logger.create_session(
            user_id="test_user", experiment_id=experiment_id
        )

        # Log an event
        event_logger.log_event(
            session_id=session_id,
            event_type=EventType.BUTTON_CLICKED,
            component="TestButton",
            action="click",
            value=json.dumps({"test": "data"}),
            duration_ms=123.45,
        )

        # Verify event was logged
        events = event_logger.get_session_events(session_id)
        # Should have 2 events: SESSION_START and BUTTON_CLICKED
        assert len(events) >= 2

        button_events = [
            e for e in events if e["event_type"] == EventType.BUTTON_CLICKED.value
        ]
        assert len(button_events) == 1

        event = button_events[0]
        assert event["component"] == "TestButton"
        assert event["action"] == "click"
        assert event["duration_ms"] == 123.45

    def test_log_negotiation_event(self, event_logger, experiment_id):
        """Test logging a negotiation-specific event."""
        session_id = event_logger.create_session(
            user_id="test_user", experiment_id=experiment_id
        )

        # Log a negotiation event
        event_logger.log_event(
            session_id=session_id,
            event_type=EventType.OFFER_ACCEPTED,
            component="Negotiation",
            scenario_id="test_scenario",
            round_number=5,
            utility_value=0.85,
            value=json.dumps({"offer": [1, 2, 3]}),
        )

        # Verify event was logged with negotiation fields
        events = event_logger.get_session_events(session_id)
        offer_events = [
            e for e in events if e["event_type"] == EventType.OFFER_ACCEPTED.value
        ]
        assert len(offer_events) == 1

        event = offer_events[0]
        assert event["scenario_id"] == "test_scenario"
        assert event["round_number"] == 5
        assert event["utility_value"] == 0.85

    def test_get_session_events(self, event_logger, experiment_id):
        """Test retrieving events for a session."""
        session_id = event_logger.create_session(
            user_id="test_user", experiment_id=experiment_id
        )

        # Log multiple events
        for i in range(5):
            event_logger.log_event(
                session_id=session_id,
                event_type=EventType.BUTTON_CLICKED,
                component=f"Button{i}",
                action="click",
            )

        # Retrieve events
        events = event_logger.get_session_events(session_id)
        # Should have 6 events: SESSION_START + 5 BUTTON_CLICKED
        assert len(events) >= 6

    def test_get_user_sessions(self, event_logger, experiment_id):
        """Test retrieving sessions for a user."""
        # Create multiple sessions for the same user
        session_id1 = event_logger.create_session(
            user_id="test_user", experiment_id=experiment_id
        )
        session_id2 = event_logger.create_session(
            user_id="test_user", experiment_id=experiment_id
        )

        # Create session for different user
        session_id3 = event_logger.create_session(
            user_id="other_user", experiment_id=experiment_id
        )

        # Get sessions for test_user
        sessions = event_logger.get_user_sessions("test_user")
        assert len(sessions) == 2

        session_ids = [s["id"] for s in sessions]
        assert session_id1 in session_ids
        assert session_id2 in session_ids
        assert session_id3 not in session_ids

    def test_get_all_users(self, event_logger, experiment_id):
        """Test retrieving all unique users."""
        # Create sessions for multiple users
        event_logger.create_session(user_id="user1", experiment_id=experiment_id)
        event_logger.create_session(user_id="user2", experiment_id=experiment_id)
        event_logger.create_session(
            user_id="user1", experiment_id=experiment_id
        )  # Duplicate user

        # Get all users
        users = event_logger.get_all_users()
        assert len(users) == 2
        assert "user1" in users
        assert "user2" in users

    def test_get_event_stats(self, event_logger, experiment_id):
        """Test getting event statistics."""
        session_id = event_logger.create_session(
            user_id="test_user", experiment_id=experiment_id
        )

        # Log events with different types
        event_logger.log_event(
            session_id=session_id,
            event_type=EventType.BUTTON_CLICKED,
            duration_ms=100.0,
        )
        event_logger.log_event(
            session_id=session_id,
            event_type=EventType.BUTTON_CLICKED,
            duration_ms=200.0,
        )
        event_logger.log_event(
            session_id=session_id, event_type=EventType.PAGE_VIEW, duration_ms=50.0
        )

        # Get stats
        stats = event_logger.get_event_stats(session_id=session_id)

        assert stats["total_events"] >= 4  # Including SESSION_START
        assert stats["avg_duration_ms"] > 0
        assert EventType.BUTTON_CLICKED.value in stats["event_counts"]
        assert stats["event_counts"][EventType.BUTTON_CLICKED.value] == 2

    def test_export_session_data(self, event_logger, experiment_id, temp_db_path):
        """Test exporting session data."""
        session_id = event_logger.create_session(
            user_id="test_user", experiment_id=experiment_id
        )

        # Log some events
        event_logger.log_event(
            session_id=session_id,
            event_type=EventType.BUTTON_CLICKED,
            component="TestButton",
        )

        # Export to file
        export_path = temp_db_path / "export.json"
        data = event_logger.export_session_data(session_id, export_path)

        # Verify export data
        assert "session" in data
        assert "events" in data
        assert "event_count" in data
        assert data["session"]["user_id"] == "test_user"
        assert data["event_count"] >= 2

        # Verify file was created
        assert export_path.exists()

        # Verify file contents
        with open(export_path) as f:
            file_data = json.load(f)
        assert file_data["session"]["user_id"] == "test_user"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_session_function(self, event_logger, experiment_id):
        """Test create_session convenience function."""
        session_id = create_session(user_id="test_user", experiment_id=experiment_id)

        assert session_id is not None
        sessions = event_logger.get_user_sessions("test_user")
        assert len(sessions) == 1

    def test_end_session_function(self, event_logger, experiment_id):
        """Test end_session convenience function."""
        session_id = create_session(user_id="test_user", experiment_id=experiment_id)
        end_session(session_id)

        sessions = event_logger.get_user_sessions("test_user")
        assert sessions[0]["is_active"] is False

    def test_log_event_function(self, event_logger, experiment_id):
        """Test log_event convenience function."""
        session_id = create_session(user_id="test_user", experiment_id=experiment_id)

        log_event(
            session_id=session_id, event_type=EventType.PAGE_VIEW, component="HomePage"
        )

        events = event_logger.get_session_events(session_id)
        page_view_events = [
            e for e in events if e["event_type"] == EventType.PAGE_VIEW.value
        ]
        assert len(page_view_events) == 1


class TestEventTypes:
    """Test different event types."""

    def test_all_event_types(self, event_logger, experiment_id):
        """Test logging all event types."""
        session_id = event_logger.create_session(
            user_id="test_user", experiment_id=experiment_id
        )

        # Log one of each event type
        for event_type in EventType:
            if event_type not in [EventType.SESSION_START, EventType.SESSION_END]:
                event_logger.log_event(
                    session_id=session_id,
                    event_type=event_type,
                    component="TestComponent",
                )

        # Verify all events were logged
        events = event_logger.get_session_events(session_id)
        event_types = {e["event_type"] for e in events}

        # Should have most event types (excluding SESSION_END which wasn't explicitly logged)
        assert len(event_types) >= len(EventType) - 2


class TestDataIntegrity:
    """Test data integrity and error handling."""

    def test_invalid_session_id(self, event_logger):
        """Test handling of invalid session ID."""
        # Try to get events for non-existent session
        events = event_logger.get_session_events("non-existent-id")
        assert events == []

    def test_dict_value_serialization(self, event_logger, experiment_id):
        """Test automatic serialization of dict values."""
        session_id = event_logger.create_session(
            user_id="test_user", experiment_id=experiment_id
        )

        # Log event with dict value
        test_dict = {"key": "value", "number": 42}
        event_logger.log_event(
            session_id=session_id,
            event_type=EventType.TOOL_INTERACTION,
            value=test_dict,
        )

        # Retrieve and verify
        events = event_logger.get_session_events(session_id)
        tool_events = [
            e for e in events if e["event_type"] == EventType.TOOL_INTERACTION.value
        ]
        assert len(tool_events) == 1

        # Value should be JSON string
        value = json.loads(tool_events[0]["value"])
        assert value == test_dict

    def test_multiple_concurrent_sessions(self, event_logger, experiment_id):
        """Test handling multiple concurrent sessions."""
        # Create multiple sessions
        sessions = []
        for i in range(5):
            session_id = event_logger.create_session(
                user_id=f"user{i}", experiment_id=experiment_id
            )
            sessions.append(session_id)

            # Log events for each session
            for j in range(3):
                event_logger.log_event(
                    session_id=session_id,
                    event_type=EventType.BUTTON_CLICKED,
                    component=f"Button{j}",
                )

        # Verify each session has its own events
        for session_id in sessions:
            events = event_logger.get_session_events(session_id)
            # Should have 4 events: SESSION_START + 3 BUTTON_CLICKED
            assert len(events) == 4


class TestPerformance:
    """Test performance with larger data sets."""

    def test_large_number_of_events(self, event_logger, experiment_id):
        """Test logging a large number of events."""
        session_id = event_logger.create_session(
            user_id="test_user", experiment_id=experiment_id
        )

        # Log many events
        num_events = 1000
        for i in range(num_events):
            event_logger.log_event(
                session_id=session_id,
                event_type=EventType.BUTTON_CLICKED,
                component=f"Button{i % 10}",
            )

        # Verify all events were logged
        events = event_logger.get_session_events(session_id)
        assert len(events) >= num_events

    def test_query_performance(self, event_logger, experiment_id):
        """Test query performance with multiple users and sessions."""
        # Create multiple users with multiple sessions each
        for i in range(10):
            for j in range(3):
                session_id = event_logger.create_session(
                    user_id=f"user{i}", experiment_id=experiment_id
                )

                # Log events
                for k in range(10):
                    event_logger.log_event(
                        session_id=session_id, event_type=EventType.BUTTON_CLICKED
                    )

        # Test query performance
        users = event_logger.get_all_users()
        assert len(users) == 10

        for user in users:
            sessions = event_logger.get_user_sessions(user)
            assert len(sessions) == 3

            for session in sessions:
                events = event_logger.get_session_events(session["id"])
                assert len(events) >= 10


class TestExperimentManagement:
    """Test experiment management functionality."""

    def test_create_experiment(self, event_logger):
        """Test creating an experiment."""
        exp_id = event_logger.create_experiment(
            name="Test Experiment 1",
            description="A test experiment",
            start_time=datetime.now(),
        )

        assert exp_id is not None
        assert isinstance(exp_id, str)

    def test_get_active_experiments(self, event_logger):
        """Test retrieving active experiments."""
        # Create multiple experiments
        exp1 = event_logger.create_experiment(
            name="Experiment 1", description="First", start_time=datetime.now()
        )
        exp2 = event_logger.create_experiment(
            name="Experiment 2", description="Second", start_time=datetime.now()
        )

        # Get active experiments
        active = event_logger.get_active_experiments()
        assert len(active) >= 2

        exp_ids = [e["id"] for e in active]
        assert exp1 in exp_ids
        assert exp2 in exp_ids

    def test_end_experiment(self, event_logger):
        """Test ending an experiment."""
        exp_id = event_logger.create_experiment(
            name="Temp Experiment", description="Temporary", start_time=datetime.now()
        )

        # End the experiment
        event_logger.end_experiment(exp_id)

        # Verify it's no longer active
        active = event_logger.get_active_experiments()
        exp_ids = [e["id"] for e in active]
        assert exp_id not in exp_ids

    def test_session_requires_experiment(self, event_logger):
        """Test that sessions must have an experiment_id."""
        exp_id = event_logger.create_experiment(
            name="Required Experiment", description="Test", start_time=datetime.now()
        )

        session_id = event_logger.create_session(
            user_id="test_user", experiment_id=exp_id
        )

        sessions = event_logger.get_user_sessions("test_user")
        assert len(sessions) == 1
        assert sessions[0]["experiment_id"] == exp_id


class TestUserManagement:
    """Test user management functionality."""

    def test_user_auto_creation(self, event_logger, experiment_id):
        """Test that users are automatically created on session creation."""
        session_id = event_logger.create_session(
            user_id="auto_user",
            experiment_id=experiment_id,
            full_name="Auto User",
            email="auto@example.com",
        )

        # Verify user was created
        user_info = event_logger.get_user_info("auto_user")
        assert user_info is not None
        assert user_info["id"] == "auto_user"
        assert user_info["full_name"] == "Auto User"
        assert user_info["email"] == "auto@example.com"
        assert user_info["login_count"] == 1

    def test_user_login_tracking(self, event_logger, experiment_id):
        """Test that user login counts are tracked."""
        # Create multiple sessions for same user
        for i in range(3):
            event_logger.create_session(
                user_id="repeat_user",
                experiment_id=experiment_id,
                full_name="Repeat User",
            )

        # Verify login count
        user_info = event_logger.get_user_info("repeat_user")
        assert user_info["login_count"] == 3

    def test_user_multiple_experiments(self, event_logger):
        """Test that users can participate in multiple experiments."""
        # Create two experiments
        exp1 = event_logger.create_experiment(
            name="Experiment A", description="First", start_time=datetime.now()
        )
        exp2 = event_logger.create_experiment(
            name="Experiment B", description="Second", start_time=datetime.now()
        )

        # Create sessions in both experiments for same user
        session1 = event_logger.create_session(
            user_id="multi_user", experiment_id=exp1, full_name="Multi User"
        )
        session2 = event_logger.create_session(
            user_id="multi_user", experiment_id=exp2, full_name="Multi User"
        )

        # Verify sessions in different experiments
        all_sessions = event_logger.get_user_sessions("multi_user")
        assert len(all_sessions) == 2

        exp_ids = [s["experiment_id"] for s in all_sessions]
        assert exp1 in exp_ids
        assert exp2 in exp_ids

        # Verify filtering by experiment
        exp1_sessions = event_logger.get_user_sessions("multi_user", experiment_id=exp1)
        assert len(exp1_sessions) == 1
        assert exp1_sessions[0]["experiment_id"] == exp1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
