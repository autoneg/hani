"""
Event tracing system for HANI.

This module provides comprehensive event tracking and session management:
- Experiment management
- User tracking
- Session tracking (login/logout)
- Negotiation event tracking (offers, accepts, rejects, etc.)
- Button click tracking across all tools
- Tool interaction tracking
- Timestamped event logs
- Database storage with SQLite

Architecture (SQLAlchemy 2.0):
- Experiment: Represents a research experiment/study
- User: Represents a user in the system
- Session: Represents a user's login session within an experiment
- Event: Represents any tracked action (button click, negotiation move, etc.)
- EventLogger: Centralized logging service
"""

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    String,
    Integer,
    DateTime,
    Float,
    Text,
    ForeignKey,
    Boolean,
    Engine,
    select,
)
from sqlalchemy.orm import (
    Session as DBSession,
    relationship,
    DeclarativeBase,
    Mapped,
    mapped_column,
)
from typing import Generator

from hani.common import DB_PATH

# Ensure database directory exists
DB_PATH.mkdir(parents=True, exist_ok=True)

# Database configuration
DATABASE_URL = f"sqlite:///{DB_PATH / 'events.db'}"


# Create SQLAlchemy base using modern DeclarativeBase
class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class EventType(str, Enum):
    """Types of events that can be tracked."""

    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Authentication events
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"

    # Scenario events
    SCENARIO_LOADED = "scenario_loaded"
    SCENARIO_STARTED = "scenario_started"

    # Negotiation events
    OFFER_MADE = "offer_made"
    OFFER_ACCEPTED = "offer_accepted"
    OFFER_REJECTED = "offer_rejected"
    NEGOTIATION_ENDED = "negotiation_ended"
    COUNTER_OFFER = "counter_offer"

    # Tool events
    TOOL_OPENED = "tool_opened"
    TOOL_CLOSED = "tool_closed"
    TOOL_INTERACTION = "tool_interaction"

    # Button events
    BUTTON_CLICKED = "button_clicked"

    # UI events
    PAGE_VIEW = "page_view"
    TAB_CHANGED = "tab_changed"
    MODAL_OPENED = "modal_opened"
    MODAL_CLOSED = "modal_closed"

    # Data events
    PREFERENCE_VIEWED = "preference_viewed"
    PLOT_VIEWED = "plot_viewed"
    RESULTS_VIEWED = "results_viewed"

    # Error events
    ERROR = "error"
    WARNING = "warning"


class Experiment(Base):
    """Represents an experiment/study."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    start_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    # Relationships
    sessions: Mapped[list["Session"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name={self.name}, active={self.is_active})>"


class User(Base):
    """Represents a user in the system."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # username
    full_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    first_login: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    login_count: Mapped[int] = mapped_column(Integer, default=0)

    # User metadata (can be extended)
    user_metadata: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON string

    # Relationships
    sessions: Mapped[list["Session"]] = relationship(back_populates="user")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, name={self.full_name}, logins={self.login_count})>"


class Session(Base):
    """Represents a user session (login to logout) within an experiment."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id"), nullable=False, index=True
    )
    experiment_id: Mapped[str] = mapped_column(
        String, ForeignKey("experiments.id"), nullable=False, index=True
    )
    start_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="sessions")
    experiment: Mapped["Experiment"] = relationship(back_populates="sessions")
    events: Mapped[list["Event"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, user={self.user_id}, experiment={self.experiment_id}, active={self.is_active})>"


class Event(Base):
    """Represents a single tracked event."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.id"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True
    )

    # Event details
    component: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    action: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Negotiation-specific fields
    scenario_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, index=True
    )
    round_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    utility_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Performance tracking
    duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationship to session
    session: Mapped["Session"] = relationship(back_populates="events")

    def __repr__(self) -> str:
        return f"<Event(id={self.id}, type={self.event_type}, time={self.timestamp})>"


class EventLogger:
    """Centralized event logging service using SQLAlchemy 2.0."""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern to ensure only one logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the event logger."""
        if self._initialized:
            return

        # Initialize database with SQLAlchemy 2.0 best practices
        self.engine: Engine = create_engine(
            DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},  # For SQLite threading
        )
        # Create tables only if they don't exist (checkfirst=True)
        Base.metadata.create_all(self.engine, checkfirst=True)

        self._initialized = True
        print(f"EventLogger initialized (database=SQLite at {DATABASE_URL})")

    @contextmanager
    def _session(self) -> Generator[DBSession, None, None]:  # type: ignore[misc]
        """Context manager for database sessions."""
        session = DBSession(self.engine)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_or_update_user(
        self,
        user_id: str,
        full_name: Optional[str] = None,
        email: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> User:
        """Create or update a user record."""
        with self._session() as db:
            user = db.scalar(select(User).where(User.id == user_id))

            if user:
                # Update existing user
                user.last_login = datetime.now(timezone.utc)
                user.login_count += 1
                if full_name:
                    user.full_name = full_name
                if email:
                    user.email = email
                if metadata:
                    user.user_metadata = json.dumps(metadata)
            else:
                # Create new user
                user = User(
                    id=user_id,
                    full_name=full_name,
                    email=email,
                    first_login=datetime.now(timezone.utc),
                    last_login=datetime.now(timezone.utc),
                    login_count=1,
                    user_metadata=json.dumps(metadata) if metadata else None,
                )
                db.add(user)

            db.commit()
            db.refresh(user)
            return user

    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        start_time: Optional[datetime] = None,
    ) -> str:
        """Create a new experiment."""
        experiment_id = str(uuid.uuid4())

        with self._session() as db:
            experiment = Experiment(
                id=experiment_id,
                name=name,
                description=description,
                start_time=start_time or datetime.now(timezone.utc),
                is_active=True,
            )
            db.add(experiment)
            return experiment_id

    def get_active_experiments(self) -> list[dict]:
        """Get all active experiments."""
        with self._session() as db:
            experiments = db.scalars(
                select(Experiment).where(Experiment.is_active == True)
            ).all()
            return [
                {
                    "id": exp.id,
                    "name": exp.name,
                    "description": exp.description,
                    "start_time": exp.start_time.isoformat(),
                    "is_active": exp.is_active,
                }
                for exp in experiments
            ]

    def end_experiment(self, experiment_id: str) -> None:
        """End an experiment by marking it inactive."""
        with self._session() as db:
            experiment = db.scalar(
                select(Experiment).where(Experiment.id == experiment_id)
            )
            if experiment:
                experiment.end_time = datetime.now(timezone.utc)
                experiment.is_active = False

    def create_session(
        self,
        user_id: str,
        experiment_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        full_name: Optional[str] = None,
        email: Optional[str] = None,
    ) -> str:
        """Create a new user session."""
        # Create or update user record
        self.create_or_update_user(user_id, full_name, email)

        session_id = str(uuid.uuid4())

        with self._session() as db:
            session = Session(
                id=session_id,
                user_id=user_id,
                experiment_id=experiment_id,
                start_time=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent,
                is_active=True,
            )
            db.add(session)

        # Log session start event
        self.log_event(
            session_id=session_id,
            event_type=EventType.SESSION_START,
            component="SessionManager",
            action="create",
            value=json.dumps({"user_id": user_id, "experiment_id": experiment_id}),
        )

        return session_id

    def end_session(self, session_id: str) -> None:
        """End a user session."""
        with self._session() as db:
            session = db.scalar(select(Session).where(Session.id == session_id))
            if session:
                session.end_time = datetime.now(timezone.utc)
                session.is_active = False

        # Log session end event
        self.log_event(
            session_id=session_id,
            event_type=EventType.SESSION_END,
            component="SessionManager",
            action="end",
        )

    def log_event(
        self,
        session_id: str,
        event_type: EventType | str,
        component: Optional[str] = None,
        action: Optional[str] = None,
        value: Optional[str | dict] = None,
        scenario_id: Optional[str] = None,
        round_number: Optional[int] = None,
        utility_value: Optional[float] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Log an event."""
        # Convert dict to JSON string
        if isinstance(value, dict):
            value = json.dumps(value)

        # Convert EventType to string
        if isinstance(event_type, EventType):
            event_type = event_type.value

        with self._session() as db:
            event = Event(
                session_id=session_id,
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                component=component,
                action=action,
                value=value,
                scenario_id=scenario_id,
                round_number=round_number,
                utility_value=utility_value,
                duration_ms=duration_ms,
            )
            db.add(event)

    def get_session_events(self, session_id: str) -> list[dict]:
        """Get all events for a session."""
        with self._session() as db:
            events = db.scalars(
                select(Event)
                .where(Event.session_id == session_id)
                .order_by(Event.timestamp)
            ).all()
            return [
                {
                    "id": e.id,
                    "event_type": e.event_type,
                    "timestamp": e.timestamp.isoformat(),
                    "component": e.component,
                    "action": e.action,
                    "value": e.value,
                    "scenario_id": e.scenario_id,
                    "round_number": e.round_number,
                    "utility_value": e.utility_value,
                    "duration_ms": e.duration_ms,
                }
                for e in events
            ]

    def get_user_sessions(
        self, user_id: str, experiment_id: Optional[str] = None
    ) -> list[dict]:
        """Get all sessions for a user, optionally filtered by experiment."""
        with self._session() as db:
            stmt = select(Session).where(Session.user_id == user_id)

            if experiment_id:
                stmt = stmt.where(Session.experiment_id == experiment_id)

            sessions = db.scalars(stmt.order_by(Session.start_time.desc())).all()

            return [
                {
                    "id": s.id,
                    "user_id": s.user_id,
                    "experiment_id": s.experiment_id,
                    "start_time": s.start_time.isoformat(),
                    "end_time": s.end_time.isoformat() if s.end_time else None,
                    "ip_address": s.ip_address,
                    "user_agent": s.user_agent,
                    "is_active": s.is_active,
                }
                for s in sessions
            ]

    def get_all_users(self) -> list[str]:
        """Get list of all unique users."""
        with self._session() as db:
            users = db.scalars(select(User)).all()
            return [u.id for u in users]

    def get_user_info(self, user_id: str) -> Optional[dict]:
        """Get detailed information about a user."""
        with self._session() as db:
            user = db.scalar(select(User).where(User.id == user_id))
            if not user:
                return None

            return {
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "first_login": user.first_login.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "login_count": user.login_count,
                "metadata": json.loads(user.user_metadata)
                if user.user_metadata
                else None,
            }

    def get_event_stats(
        self, session_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> dict:
        """Get statistics about events."""
        with self._session() as db:
            stmt = select(Event)

            if session_id:
                stmt = stmt.where(Event.session_id == session_id)
            elif user_id:
                # Get all sessions for user
                session_ids = [
                    s.id
                    for s in db.scalars(
                        select(Session).where(Session.user_id == user_id)
                    ).all()
                ]
                stmt = stmt.where(Event.session_id.in_(session_ids))

            events = db.scalars(stmt).all()

            # Count by event type
            event_counts: dict[str, int] = {}
            for event in events:
                event_counts[event.event_type] = (
                    event_counts.get(event.event_type, 0) + 1
                )

            # Calculate average duration
            durations = [e.duration_ms for e in events if e.duration_ms is not None]
            avg_duration = sum(durations) / len(durations) if durations else 0

            return {
                "total_events": len(events),
                "event_counts": event_counts,
                "avg_duration_ms": avg_duration,
                "first_event": events[0].timestamp.isoformat() if events else None,
                "last_event": events[-1].timestamp.isoformat() if events else None,
            }

    def export_session_data(
        self, session_id: str, output_path: Optional[Path] = None
    ) -> dict:
        """Export complete session data including all events."""
        with self._session() as db:
            session = db.scalar(select(Session).where(Session.id == session_id))
            if not session:
                return {"error": "Session not found"}

            events = self.get_session_events(session_id)

            data = {
                "session": {
                    "id": session.id,
                    "user_id": session.user_id,
                    "experiment_id": session.experiment_id,
                    "start_time": session.start_time.isoformat(),
                    "end_time": session.end_time.isoformat()
                    if session.end_time
                    else None,
                    "is_active": session.is_active,
                    "ip_address": session.ip_address,
                    "user_agent": session.user_agent,
                },
                "events": events,
                "event_count": len(events),
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }

        # Save to file if path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        return data


# Global logger instance
_event_logger = None


def get_event_logger() -> EventLogger:
    """Get the global event logger instance."""
    global _event_logger
    if _event_logger is None:
        _event_logger = EventLogger()
    return _event_logger


def log_event(session_id: str, event_type: EventType | str, **kwargs) -> None:
    """Log an event (convenience function)."""
    logger = get_event_logger()
    logger.log_event(session_id, event_type, **kwargs)


def create_session(user_id: str, experiment_id: str, **kwargs) -> str:
    """Convenience function to create a session."""
    logger = get_event_logger()
    return logger.create_session(user_id, experiment_id, **kwargs)


def end_session(session_id: str) -> None:
    """End a session (convenience function)."""
    logger = get_event_logger()
    logger.end_session(session_id)


def create_experiment(name: str, description: Optional[str] = None, **kwargs) -> str:
    """Convenience function to create an experiment."""
    logger = get_event_logger()
    return logger.create_experiment(name, description, **kwargs)


def get_active_experiments() -> list[dict]:
    """Convenience function to get active experiments."""
    logger = get_event_logger()
    return logger.get_active_experiments()


def end_experiment(experiment_id: str) -> None:
    """End an experiment (convenience function)."""
    logger = get_event_logger()
    logger.end_experiment(experiment_id)
