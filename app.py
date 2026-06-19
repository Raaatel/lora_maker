"""LoRA Maker - Entry point."""

import argparse
import webbrowser
import threading
import uvicorn


def open_browser(port: int):
    import time
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{port}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Maker")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    print(f"\n  LoRA Maker")
    print(f"  http://localhost:{args.port}\n")

    uvicorn.run(
        "server.app_factory:create_app",
        host=args.host,
        port=args.port,
        factory=True,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
