import sys
import subprocess
from pathlib import Path

from hani.common import (
    REG_PORT,
    LOGIN_FILE,
    OAUTH_PROVIDER,
    OAUTH_KEY,
    OAUTH_SECRET,
    OAUTH_REDIRECT_URI,
    OAUTH_ENCRYPTION_KEY,
    COOKIE_SECRET,
)


def main():
    # Determine authentication mode
    from hani.auth import get_auth_mode, create_hashed_users_file

    auth_mode = get_auth_mode()

    print(f"🔐 Registration app authentication mode: {auth_mode.upper()}")

    # Build base command
    base_cmd = [
        "panel",
        "serve",
        str(Path(__file__).parent / "register.py"),
        "--port",
        str(REG_PORT),
    ]

    # Add dev flag if requested
    dev_args = (
        ["--dev"]
        if len(sys.argv) > 1 and any(_ in sys.argv[1:] for _ in ("dev",))
        else []
    )

    # Add extra user args (excluding our special keywords)
    extra_args = (
        [_ for _ in sys.argv[1:] if _ not in ("dev", "login", "port")]
        if len(sys.argv) > 1
        else []
    )

    if auth_mode == "oauth":
        # OAuth mode - use GitHub/Google/etc authentication
        # Note: Registration app typically doesn't require auth itself,
        # but we configure it to match the main app for consistency
        print(f"  Provider: {OAUTH_PROVIDER}")
        print(f"  Redirect URI: {OAUTH_REDIRECT_URI}")

        if not OAUTH_KEY or not OAUTH_SECRET:
            print(
                "⚠️  OAuth credentials not configured - running without authentication"
            )
            print("   (This is normal for a registration app)")
            auth_args = []
        elif not OAUTH_ENCRYPTION_KEY:
            print(
                "⚠️  OAuth encryption key not configured - running without authentication"
            )
            auth_args = []
        else:
            # Full OAuth configuration
            auth_args = [
                "--oauth-provider",
                OAUTH_PROVIDER,
                "--oauth-key",
                OAUTH_KEY,
                "--oauth-secret",
                OAUTH_SECRET,
                "--oauth-redirect-uri",
                OAUTH_REDIRECT_URI,
                "--oauth-encryption-key",
                OAUTH_ENCRYPTION_KEY,
                "--cookie-secret",
                COOKIE_SECRET,
            ]
            print("✓ OAuth authentication configured")

    else:
        # Password mode - registration app typically doesn't need auth
        # but we set up cookie handling for consistency
        print(f"  Password file location: {LOGIN_FILE}")

        # Check if we need to convert plain text passwords to hashed
        hashed_file = LOGIN_FILE.parent / "users_hashed.json"

        if not hashed_file.exists() and LOGIN_FILE.exists():
            print("⚠️  Converting plain text passwords to hashed format...")
            # Create hashed version
            create_hashed_users_file(LOGIN_FILE, hashed_file)
            print(f"✓ Created hashed password file: {hashed_file}")

        # Registration app doesn't require login, but we set up cookies
        # for session consistency with main app
        auth_args = [
            "--cookie-secret",
            COOKIE_SECRET,
        ]

        print(
            "✓ Password mode configured (no authentication required for registration)"
        )

    # Build final command
    final_cmd = base_cmd + dev_args + auth_args + extra_args

    print(f"\n🚀 Starting HANI Registration server on port {REG_PORT}...\n")

    try:
        subprocess.run(final_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Panel app: {e}")


if __name__ == "__main__":
    main()
