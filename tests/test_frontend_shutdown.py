"""Pins the browser shutdown ordering with a deterministic Node model.

The model (frontend_shutdown.test.mjs) reproduces the stop race in app.js and
asserts the fixed ordering: all captured audio - including the in-flight final
block and the pcmCarry tail - is placed on the socket as full-size frames
before the stop control, which must be the last message.
"""

import pathlib
import shutil
import subprocess

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

_SCRIPT = pathlib.Path(__file__).with_name("frontend_shutdown.test.mjs")


def test_browser_stop_flushes_audio_before_the_stop_control() -> None:
    result = subprocess.run(
        [NODE, str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"model assertion failed:\n{result.stdout}\n{result.stderr}"
