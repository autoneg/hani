#!/usr/bin/env python3
"""
Custom HANI server launcher with dual authentication support.

This module starts the HANI Panel application with both password and OAuth
authentication enabled. It uses Panel's serve() function directly to add
custom OAuth handlers via extra_patterns.
"""

import os
import sys
from pathlib import Path

import panel as pn

# Add the src directory to path if running from project root
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir.parent))


def serve_with_dual_auth(
    port: int = 5006,
    address: str = "0.0.0.0",
    show: bool = False,
    dev: bool = False,
    **kwargs,
):
    """
    Start the HANI server with dual authentication support.

    Args:
        port: Port to serve on
        address: Address to bind to
        show: Whether to open browser automatically
        dev: Whether to run in development mode with auto-reload
        **kwargs: Additional arguments passed to pn.serve()
    """
    from hani.common import (
        USERS_FILE,
        COOKIE_SECRET,
        OAUTH_KEY,
        OAUTH_SECRET,
    )
    from hani.auth import ensure_admin_user

    # Ensure admin user exists
    ensure_admin_user()

    # Import the app module
    from hani import app as hani_app

    # Build extra patterns for OAuth if enabled
    extra_patterns = list(kwargs.pop("extra_patterns", []))

    if OAUTH_KEY and OAUTH_SECRET:
        from hani.dual_auth import get_oauth_handlers

        oauth_handlers = get_oauth_handlers()
        extra_patterns.extend(oauth_handlers)
        print(f"✓ OAuth handlers added: {[h[0] for h in oauth_handlers]}")

    # Configure templates
    template_dir = Path(__file__).parent / "templates"
    login_template = str(template_dir / "basic_login.html")
    logout_template = str(template_dir / "logout.html")

    # Enable dual auth mode
    os.environ["_HANI_DUAL_AUTH"] = "1"

    print(f"Starting HANI server on http://{address}:{port}")
    print(f"  Authentication: Dual (password + OAuth)")
    print(f"  Password file: {USERS_FILE}")

    # Use Panel's serve function with our custom patterns
    pn.serve(
        {"": hani_app},  # Serve at root
        port=port,
        address=address,
        show=show,
        basic_auth=str(USERS_FILE),
        cookie_secret=COOKIE_SECRET,
        login_template=login_template,
        logout_template=logout_template,
        extra_patterns=extra_patterns,
        websocket_origin=["*"],  # Allow all origins for development
        **kwargs,
    )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run HANI with dual authentication")
    parser.add_argument("--port", type=int, default=5006, help="Port to serve on")
    parser.add_argument("--address", default="0.0.0.0", help="Address to bind to")
    parser.add_argument(
        "--show", action="store_true", help="Open browser automatically"
    )
    parser.add_argument(
        "--dev", action="store_true", help="Development mode with auto-reload"
    )

    args = parser.parse_args()

    serve_with_dual_auth(
        port=args.port,
        address=args.address,
        show=args.show,
        dev=args.dev,
    )


if __name__ == "__main__":
    main()
