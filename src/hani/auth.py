"""
User authentication utilities for Panel.

This module provides:
- OAuth configuration detection
- Admin user management via env.json configuration
- User consent tracking

User data is stored in two files:
- users.json: Panel's expected format {"username": "password", ...}
- users_meta.json: Additional metadata {"username": {"consented": true, "consent_date": "...", ...}, ...}
"""

from __future__ import annotations
import json
from typing import Literal
from datetime import datetime


def get_auth_mode() -> Literal["password", "oauth"]:
    """Determine which authentication mode to use.

    Returns:
        'oauth' if OAuth credentials are configured, otherwise 'password'
    """
    from hani.common import AUTH_MODE, OAUTH_KEY, OAUTH_SECRET

    auth_mode = (AUTH_MODE or "auto").lower()

    if auth_mode == "oauth":
        return "oauth"
    elif auth_mode == "password":
        return "password"
    else:  # auto mode
        # Use OAuth if credentials are provided
        if OAUTH_KEY and OAUTH_SECRET:
            return "oauth"
        return "password"


def load_users() -> dict:
    """Load all users from users.json.

    Returns:
        Dict of username -> password (plain text)
    """
    from hani.common import USERS_FILE

    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_users(users: dict) -> None:
    """Save all users to users.json.

    Args:
        users: Dict of username -> password
    """
    from hani.common import USERS_FILE, SETTINGS_DIR

    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def load_users_meta() -> dict:
    """Load user metadata from users_meta.json.

    Returns:
        Dict of username -> {consented: bool, consent_date: str, ...}
    """
    from hani.common import SETTINGS_DIR

    meta_file = SETTINGS_DIR / "users_meta.json"
    if not meta_file.exists():
        return {}
    try:
        with open(meta_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_users_meta(meta: dict) -> None:
    """Save user metadata to users_meta.json.

    Args:
        meta: Dict of username -> metadata
    """
    from hani.common import SETTINGS_DIR

    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    meta_file = SETTINGS_DIR / "users_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)


def get_user_consent(username: str) -> bool:
    """Check if user has consented.

    Args:
        username: Username to check

    Returns:
        True if user has consented, False otherwise
    """
    meta = load_users_meta()
    user_meta = meta.get(username, {})
    return user_meta.get("consented", False)


def set_user_consent(username: str, consented: bool = True, **extra_fields) -> bool:
    """Set user consent status.

    Args:
        username: Username to update
        consented: Whether user has consented
        **extra_fields: Additional fields to store (name, consent_date, etc.)

    Returns:
        True if updated successfully
    """
    meta = load_users_meta()
    if username not in meta:
        meta[username] = {}

    meta[username]["consented"] = consented
    if consented:
        meta[username]["consent_date"] = extra_fields.get(
            "consent_date", datetime.now().strftime("%Y-%m-%d")
        )

    # Store any extra fields
    for key, value in extra_fields.items():
        if key != "consent_date":
            meta[username][key] = value

    save_users_meta(meta)
    return True


def ensure_default_users() -> None:
    """Ensure default users (admin and user) exist in users.json.

    This function creates default users only if they don't already exist.
    - admin: password from admin.password in env.json, defaults to 'adminpass'
    - user: regular user with password 'userpass'

    Passwords are stored in plain text (Panel's expected format).
    Default users are marked as consented.
    """
    from hani.common import SETTINGS_DIR, ADMIN_PASS

    # Ensure settings directory exists
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

    users = load_users()
    meta = load_users_meta()
    created_users = []

    # Create admin if it doesn't exist
    if "admin" not in users:
        admin_password = ADMIN_PASS if ADMIN_PASS else "adminpass"
        users["admin"] = admin_password
        meta["admin"] = {"consented": True, "consent_date": "default"}
        created_users.append("admin")

    # Create regular user if it doesn't exist
    if "user" not in users:
        users["user"] = "userpass"
        meta["user"] = {"consented": True, "consent_date": "default"}
        created_users.append("user")

    if created_users:
        save_users(users)
        save_users_meta(meta)
        print(f"Created default users: {', '.join(created_users)}")
    else:
        print("Default users already exist")


# Alias for backward compatibility
ensure_admin_user = ensure_default_users


def get_user(username: str) -> dict | None:
    """Get user data by username.

    Args:
        username: Username to look up

    Returns:
        User data dict or None if not found
    """
    users = load_users()
    if username not in users:
        return None

    return {"username": username}


def create_user(
    username: str,
    password: str,
    **extra_fields,
) -> bool:
    """Create a new user.

    Args:
        username: Username for the new user
        password: Password (stored as plain text)
        **extra_fields: Ignored (for API compatibility)

    Returns:
        True if user was created, False if username already exists
    """
    users = load_users()

    if username in users:
        return False

    users[username] = password
    save_users(users)
    return True


def update_user_password(username: str, new_password: str) -> bool:
    """Update a user's password.

    Args:
        username: Username whose password to update
        new_password: New password

    Returns:
        True if password was updated, False if user not found
    """
    users = load_users()

    if username not in users:
        return False

    users[username] = new_password
    save_users(users)
    return True


def verify_user_password(username: str, password: str) -> bool:
    """Verify a username/password combination.

    Args:
        username: Username to check
        password: Password to verify

    Returns:
        True if password matches, False otherwise
    """
    users = load_users()

    if username not in users:
        return False

    return users[username] == password


def update_user(username: str, **fields) -> bool:
    """Update user fields (no-op in simplified auth, kept for API compatibility).

    In the simplified auth system, users.json only stores username -> password.
    This function exists for API compatibility but doesn't actually store extra fields.

    Args:
        username: Username to update
        **fields: Fields to update (ignored in simplified auth)

    Returns:
        True if user exists, False otherwise
    """
    users = load_users()
    return username in users


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python auth.py <command> [args]")
        print("\nCommands:")
        print("  ensure-admin          Create default users if not exist")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ensure-admin":
        ensure_admin_user()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
