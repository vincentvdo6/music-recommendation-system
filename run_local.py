#!/usr/bin/env python3
"""Simple startup script for the music recommendation system."""

import os
import subprocess
import sys
import threading
import time
import webbrowser


def open_browser_delayed(url: str):
    """Open browser after server starts."""
    time.sleep(3)
    try:
        webbrowser.open(url)
        print("Browser opened automatically!")
    except Exception:
        print(f"Please open {url} in your browser")


def configured_port() -> int:
    try:
        port = int(os.getenv("PORT", "8000"))
    except ValueError as exc:
        raise SystemExit("PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise SystemExit("PORT must be between 1 and 65535")
    return port


def main():
    port = configured_port()
    url = f"http://localhost:{port}"
    print("Music Recommendation System")
    print("=" * 30)
    print()

    # Open browser in background
    browser_thread = threading.Thread(target=open_browser_delayed, args=(url,), daemon=True)
    browser_thread.start()

    print("Starting server...")
    print(f"   Website: {url}")
    print(f"   API Docs: {url}/docs")
    print(f"   Health: {url}/health")
    print()
    print("Press Ctrl+C to stop")
    print()

    try:
        # Start uvicorn server
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--reload",
            "--no-access-log",
        ]

        log_level = os.getenv("UVICORN_LOG_LEVEL")
        if log_level:
            cmd.extend(["--log-level", log_level.lower()])

        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nServer stopped")


if __name__ == "__main__":
    main()
