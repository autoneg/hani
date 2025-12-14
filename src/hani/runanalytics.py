#!/usr/bin/env python
"""
Entry point for the HANI Event Analytics Dashboard.

This script launches the Panel-based analytics dashboard for viewing
event tracking data collected from HANI sessions.
"""

import sys
import subprocess


def main():
    """Launch the event analytics dashboard."""
    try:
        from pathlib import Path
        import hani.event_analytics

        # Get the path to the event_analytics module
        analytics_path = Path(hani.event_analytics.__file__)

        print("🚀 Launching HANI Event Analytics Dashboard...")
        print(f"📊 Dashboard: {analytics_path}")
        print("🌐 Opening in browser...")
        print()
        print("Press Ctrl+C to stop the server")
        print()

        # Launch panel serve with the analytics dashboard
        subprocess.run(["panel", "serve", str(analytics_path), "--show"], check=True)
    except KeyboardInterrupt:
        print("\n\n✓ Analytics dashboard stopped")
        sys.exit(0)
    except FileNotFoundError:
        print("❌ Error: 'panel' command not found. Please ensure Panel is installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching analytics dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
