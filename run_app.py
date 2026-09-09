"""AquaSense AI — Production Web Application Server Launcher."""
import os
import sys
import time
import webbrowser
import threading
import uvicorn
import urllib.request

APP_HOST = "127.0.0.1"
APP_PORT = 8000
APP_URL = f"http://{APP_HOST}:{APP_PORT}"


def open_browser_delayed():
    """Waits for server to start, then opens the default browser."""
    time.sleep(1.8)
    try:
        webbrowser.open(APP_URL)
    except Exception:
        pass


def main():
    print("=" * 65)
    print("      💧 AquaSense AI — Intelligent Water Quality System")
    print("=" * 65)
    print(f"\n🚀 Starting Web Application Server on {APP_URL} ...")
    print(f"📚 API Documentation: {APP_URL}/docs")
    print("-" * 65)
    print("💡 Press Ctrl+C to terminate the server.\n")

    # Open browser automatically in background thread
    threading.Thread(target=open_browser_delayed, daemon=True).start()

    # Run Uvicorn directly in this process (single terminal window!)
    uvicorn.run(
        "src.api.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
