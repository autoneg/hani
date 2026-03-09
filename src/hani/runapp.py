import sys
import subprocess
from pathlib import Path
import argparse
import os

from hani.common import (
    LOGIN_FILE,
    OAUTH_PROVIDER,
    OAUTH_KEY,
    OAUTH_SECRET,
    OAUTH_REDIRECT_URI,
    OAUTH_ENCRYPTION_KEY,
    COOKIE_SECRET,
)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run HANI application")
    parser.add_argument(
        "--agents",
        type=str,
        help="Comma-separated list of negotiator types (e.g., 'AspirationNegotiator,helpers.AgentK,LLMHybridNegotiator')",
    )
    # Parse known args to allow panel's args to pass through
    args, unknown_args = parser.parse_known_args()

    # If --agents is provided via command-line (not from run.py), set it as environment variable
    # run.py already sets _HANI_CMDLINE_AGENTS, so we only set it if not already set
    if args.agents and not os.environ.get("_HANI_CMDLINE_AGENTS"):
        print(f"🤖 Using agent types: {args.agents}")
        os.environ["_HANI_CMDLINE_AGENTS"] = args.agents
    elif os.environ.get("_HANI_CMDLINE_AGENTS"):
        print(f"🤖 Using agent types: {os.environ['_HANI_CMDLINE_AGENTS']}")

    # Determine authentication mode
    from hani.auth import get_auth_mode, create_hashed_users_file, ensure_admin_user

    auth_mode = get_auth_mode()

    print(f"🔐 Authentication mode: {auth_mode.upper()}")

    # Build base command
    base_cmd = [
        "panel",
        "serve",
        str(Path(__file__).parent / "app.py"),
    ]

    # Add templates
    template_args = [
        "--logout-template",
        "src/hani/templates/logout.html",
        "--basic-login-template",
        "src/hani/templates/basic_login.html",
    ]

    # Add dev flag if requested
    dev_args = (
        ["--dev"]
        if len(sys.argv) > 1 and any(_ in sys.argv[1:] for _ in ("dev",))
        else []
    )

    # Add extra user args (excluding our special keywords and --agents)
    excluded_keywords = ["dev", "login", "port", "--agents"]
    extra_args = []
    if len(sys.argv) > 1:
        skip_next = False
        for arg in unknown_args:
            if skip_next:
                skip_next = False
                continue
            if arg in excluded_keywords:
                # Skip this arg and potentially the next one
                if arg == "--agents":
                    skip_next = True
                continue
            extra_args.append(arg)

    if auth_mode == "oauth":
        # OAuth mode - use GitHub/Google/etc authentication
        print(f"  Provider: {OAUTH_PROVIDER}")
        print(f"  Redirect URI: {OAUTH_REDIRECT_URI}")

        if not OAUTH_KEY or not OAUTH_SECRET:
            print("❌ ERROR: OAuth credentials not configured!")
            print("   Set HANI_OAUTH_KEY and HANI_OAUTH_SECRET environment variables")
            print("   Or use password authentication (unset HANI_OAUTH_KEY)")
            sys.exit(1)

        if not OAUTH_ENCRYPTION_KEY:
            print("❌ ERROR: OAuth encryption key not configured!")
            print("   Set HANI_OAUTH_ENCRYPTION_KEY environment variable")
            print(
                '   Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
            sys.exit(1)

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
        # Password mode - use hashed password file
        print(f"  Using password file: {LOGIN_FILE}")

        # Ensure admin user exists with password from ADMIN_PASS env var
        ensure_admin_user()

        # Check if we need to convert plain text passwords to hashed
        hashed_file = LOGIN_FILE.parent / "users_hashed.json"
        plain_backup = LOGIN_FILE.parent / "users_plain_backup.json"

        if not hashed_file.exists() and LOGIN_FILE.exists():
            print("⚠️  Converting plain text passwords to hashed format...")
            # Create backup of plain text file
            import shutil

            shutil.copy(LOGIN_FILE, plain_backup)
            print(f"✓ Backed up plain text passwords to {plain_backup}")

            # Create hashed version
            create_hashed_users_file(LOGIN_FILE, hashed_file)
            print(f"✓ Created hashed password file: {hashed_file}")

        # Use hashed file if it exists, otherwise fall back to plain
        # Note: Panel authentication patching happens in app.py when the app is loaded
        auth_file = hashed_file if hashed_file.exists() else LOGIN_FILE

        auth_args = [
            "--basic-auth",
            str(auth_file),
            "--cookie-secret",
            COOKIE_SECRET,
        ]

        print("✓ Password authentication configured")

    # Build final command (no agents_args needed - passed via environment variable)
    final_cmd = base_cmd + template_args + dev_args + auth_args + extra_args

    print(f"\n🚀 Starting HANI server...\n")

    try:
        subprocess.run(final_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Panel app: {e}")


if __name__ == "__main__":
    main()
