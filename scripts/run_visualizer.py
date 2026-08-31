"""Run the local LeRobot dataset visualizer.

Starts a local-only Flask app (Werkzeug's built-in dev server -- single-user,
local, not for concurrent or production use) serving the read-only JSON API and
static frontend over the confidential dataset. Binds to 127.0.0.1 by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot_visualizer.webapp import create_app  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, local-only).")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug/reload mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.dataset_root)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
