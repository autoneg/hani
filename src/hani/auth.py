"""
User authentication utilities with password hashing and OAuth support.

This module provides:
- Secure password hashing for Panel authentication
- OAuth configuration detection
- Dual authentication mode support
- Admin user management via ADMIN_PASS environment variable

User data is stored in a single users.json file with structure:
{
    "username": {
        "password": "<hashed_password>",
        "email": "...",
        "name": "...",
        ...other profile fields...
    }
}
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Literal


def get_auth_mode() -> Literal["password", "oauth"]:
    """Determine which authentication mode to use.

    Returns:
        'oauth' if OAuth credentials are configured, otherwise 'password'
    """
    from hani.common import AUTH_MODE, OAUTH_KEY, OAUTH_SECRET

    if AUTH_MODE.lower() == "oauth":
        return "oauth"
    elif AUTH_MODE.lower() == "password":
        return "password"
    else:  # auto mode
        # Use OAuth if credentials are provided
        if OAUTH_KEY and OAUTH_SECRET:
            return "oauth"
        return "password"


def hash_password(password: str, salt: str = "hani-secure-salt-2025") -> str:
    """Hash a password using SHA256 with salt.

    Args:
        password: Plain text password
        salt: Salt to use for hashing

    Returns:
        Hex digest of hashed password
    """
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def load_users() -> dict:
    """Load all users from users.json.

    Returns:
        Dict of username -> user data (including hashed password)
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
        users: Dict of username -> user data (including hashed password)
    """
    from hani.common import USERS_FILE, SETTINGS_DIR

    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def ensure_admin_user(salt: str = "hani-secure-salt-2025") -> None:
    """Ensure admin user exists in users.json.

    This function creates the admin user only if it doesn't already exist.
    The password is taken from the ADMIN_PASS environment variable, defaulting to 'adminpass'.
    Only hashed passwords are stored.

    Args:
        salt: Salt to use for password hashing
    """
    from hani.common import SETTINGS_DIR, ADMIN_PASS

    # Ensure settings directory exists
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

    users = load_users()

    # Only create admin if it doesn't already exist
    if "admin" in users:
        print("✓ Admin user already exists (keeping existing password)")
        return

    # Create admin user with hashed password
    users["admin"] = {
        "password": hash_password(ADMIN_PASS, salt),
        "email": "admin@hani.local",
        "name": "Administrator",
        "is_admin": True,
    }

    save_users(users)

    print(
        f"✓ Admin user created (password from {'ADMIN_PASS env var' if os.getenv('ADMIN_PASS') else 'default'})"
    )


def verify_user_password(
    username: str,
    password: str,
    salt: str = "hani-secure-salt-2025",
) -> bool:
    """Verify a username/password against users.json.

    Args:
        username: Username to check
        password: Plain text password to verify
        salt: Salt used for hashing

    Returns:
        True if password matches, False otherwise
    """
    users = load_users()

    if username not in users:
        return False

    user_data = users[username]
    stored_hash = user_data.get("password", "")

    return stored_hash == hash_password(password, salt)


def get_user(username: str) -> dict | None:
    """Get user data by username.

    Args:
        username: Username to look up

    Returns:
        User data dict (without password) or None if not found
    """
    users = load_users()
    if username not in users:
        return None

    user_data = users[username].copy()
    # Don't expose password hash
    user_data.pop("password", None)
    return user_data


def create_user(
    username: str,
    password: str,
    email: str = "",
    name: str = "",
    salt: str = "hani-secure-salt-2025",
    **extra_fields,
) -> bool:
    """Create a new user with hashed password.

    Args:
        username: Username for the new user
        password: Plain text password (will be hashed)
        email: User's email
        name: User's full name
        salt: Salt for password hashing
        **extra_fields: Any additional user profile fields

    Returns:
        True if user was created, False if username already exists
    """
    users = load_users()

    if username in users:
        return False

    users[username] = {
        "password": hash_password(password, salt),
        "email": email,
        "name": name,
        **extra_fields,
    }

    save_users(users)
    return True


def update_user(username: str, **fields) -> bool:
    """Update user profile fields (not password).

    Args:
        username: Username to update
        **fields: Fields to update (password cannot be updated this way)

    Returns:
        True if user was updated, False if user not found
    """
    users = load_users()

    if username not in users:
        return False

    # Don't allow password updates through this function
    fields.pop("password", None)

    users[username].update(fields)
    save_users(users)
    return True


def update_user_password(
    username: str, new_password: str, salt: str = "hani-secure-salt-2025"
) -> bool:
    """Update a user's password.

    Args:
        username: Username whose password to update
        new_password: New plain text password (will be hashed)
        salt: Salt for password hashing

    Returns:
        True if password was updated, False if user not found
    """
    users = load_users()

    if username not in users:
        return False

    users[username]["password"] = hash_password(new_password, salt)
    save_users(users)
    return True


def patch_panel_auth(salt: str = "hani-secure-salt-2025"):
    """Monkey-patch Panel's BasicLoginHandler to use hashed passwords from users.json.

    This must be called before starting the Panel server.
    The users.json file contains user objects with a 'password' field that stores the hash.

    Args:
        salt: Salt to use for password hashing
    """
    try:
        from panel.auth import BasicLoginHandler
        from panel.io.state import state
        from panel.config import config
    except ImportError:
        print("Warning: Could not import Panel auth modules")
        return

    # Store original _validate method
    original_validate = BasicLoginHandler._validate

    def hashed_validate(self, username, password):
        """Validate username/password against hashed passwords in users.json."""
        if "basic_auth" in state._server_config.get(self.application, {}):
            auth_info = state._server_config[self.application]["basic_auth"]
        else:
            auth_info = config.basic_auth

        if isinstance(auth_info, str) and os.path.isfile(auth_info):
            with open(auth_info, encoding="utf-8") as auth_file:
                auth_info = json.loads(auth_file.read())

        if isinstance(auth_info, dict):
            if username not in auth_info:
                # Log failed login attempt
                try:
                    from hani.events import EventType, log_event
                    from hani.event_tracking import get_current_session_id

                    session_id = get_current_session_id()
                    if session_id:
                        log_event(
                            session_id=session_id,
                            event_type=EventType.LOGIN_FAILED,
                            component="Auth",
                            action="login",
                            value=json.dumps(
                                {"username": username, "reason": "user_not_found"}
                            ),
                        )
                except:
                    pass
                return False

            # Get the stored password hash
            # Support both formats:
            # 1. Simple: {"username": "hash"}
            # 2. Full user object: {"username": {"password": "hash", ...}}
            user_entry = auth_info[username]
            if isinstance(user_entry, dict):
                stored_hash = user_entry.get("password", "")
            else:
                stored_hash = user_entry

            # Compare hashed password
            password_hash = hash_password(password, salt)
            success = password_hash == stored_hash

            # Log login attempt
            try:
                from hani.events import EventType, log_event
                from hani.event_tracking import get_current_session_id

                session_id = get_current_session_id()
                if session_id:
                    if success:
                        log_event(
                            session_id=session_id,
                            event_type=EventType.LOGIN,
                            component="Auth",
                            action="login",
                            value=json.dumps({"username": username}),
                        )
                    else:
                        log_event(
                            session_id=session_id,
                            event_type=EventType.LOGIN_FAILED,
                            component="Auth",
                            action="login",
                            value=json.dumps(
                                {"username": username, "reason": "invalid_password"}
                            ),
                        )
            except:
                pass

            return success
        elif isinstance(auth_info, str):
            # Single password case - hash and compare
            password_hash = hash_password(password, salt)
            return password_hash == auth_info
        return False

    # Replace the _validate method
    BasicLoginHandler._validate = hashed_validate
    print("✓ Panel authentication patched to use hashed passwords")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python auth.py <command> [args]")
        print("\nCommands:")
        print("  ensure-admin          Create admin user if not exists")
        print("  hash <password>       Hash a password")
        print("  verify <user> <pass>  Verify a user's password")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ensure-admin":
        ensure_admin_user()
    elif command == "hash" and len(sys.argv) > 2:
        print(hash_password(sys.argv[2]))
    elif command == "verify" and len(sys.argv) > 3:
        result = verify_user_password(sys.argv[2], sys.argv[3])
        print(f"Password valid: {result}")
    else:
        print(f"Unknown command or missing arguments: {command}")
        sys.exit(1)
