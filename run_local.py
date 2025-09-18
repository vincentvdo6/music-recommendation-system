#!/usr/bin/env python3
"""Simple startup script for the music recommendation system."""

import subprocess
import sys
import webbrowser
import time
import threading


def open_browser_delayed():
    """Open browser after server starts."""
    time.sleep(3)
    try:
        webbrowser.open("http://localhost:8000")
        print("Browser opened automatically!")
    except:
        print("Please open http://localhost:8000 in your browser")


def main():
    print("Music Recommendation System")
    print("=" * 30)
    print()
    
    # Open browser in background
    browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
    browser_thread.start()
    
    print("Starting server...")
    print("   Website: http://localhost:8000") 
    print("   API Docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\nServer stopped")


if __name__ == "__main__":
    main()