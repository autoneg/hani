"""
User authentication utilities with password hashing and OAuth support.

This module provides:
- Secure password hashing for Panel authentication
- OAuth configuration detection
- Dual authentication mode support
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


def create_hashed_users_file(
    plain_users_file: Path | str,
    hashed_users_file: Path | str,
    salt: str = "hani-secure-salt-2025",
) -> None:
    """Convert a plain text users.json file to hashed passwords.

    Args:
        plain_users_file: Path to file with plain passwords
        hashed_users_file: Path to output file with hashed passwords
        salt: Salt to use for hashing
    """
    with open(plain_users_file) as f:
        plain_users = json.load(f)

    hashed_users = {
        username: hash_password(password, salt)
        for username, password in plain_users.items()
    }

    with open(hashed_users_file, "w") as f:
        json.dump(hashed_users, f, indent=2)

    print(f"Created hashed users file at {hashed_users_file}")
    print(f"Converted {len(hashed_users)} users")


def verify_password(
    username: str,
    password: str,
    users_file: Path | str,
    salt: str = "hani-secure-salt-2025",
) -> bool:
    """Verify a username/password against hashed users file.

    Args:
        username: Username to check
        password: Plain text password to verify
        users_file: Path to hashed users file
        salt: Salt used for hashing

    Returns:
        True if password matches, False otherwise
    """
    with open(users_file) as f:
        users = json.load(f)

    if username not in users:
        return False

    return users[username] == hash_password(password, salt)


def patch_panel_auth(salt: str = "hani-secure-salt-2025"):
    """Monkey-patch Panel's BasicLoginHandler to use hashed passwords.

    This must be called before starting the Panel server.

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
        """Validate username/password against hashed passwords."""
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
            # Compare hashed password
            password_hash = hash_password(password, salt)
            success = password_hash == auth_info[username]

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
        print("Usage: python auth.py <plain_users_file> [hashed_users_file]")
        print("\nConverts a plain text users.json to hashed passwords")
        sys.exit(1)

    plain_file = Path(sys.argv[1])
    if len(sys.argv) > 2:
        hashed_file = Path(sys.argv[2])
    else:
        hashed_file = plain_file.parent / "users_hashed.json"

    create_hashed_users_file(plain_file, hashed_file)
