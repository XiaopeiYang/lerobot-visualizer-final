"""Frontend synchronization regression tests executed with Node when available."""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


class FrontendSynchronizationTests(unittest.TestCase):
    def _run_node_test(self, filename: str) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is not installed")

        test_file = Path(__file__).parent / "js" / filename
        completed = subprocess.run(
            [node, "--test", str(test_file)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_latest_frame_queue(self) -> None:
        self._run_node_test("latestFrameQueue.test.mjs")

    def test_video_panel_sync(self) -> None:
        self._run_node_test("videoPanel.test.mjs")


if __name__ == "__main__":
    unittest.main()
