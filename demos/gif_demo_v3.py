"""Vectro v3.0.0 — GIF recording wrapper.

Patches time.sleep to a short cap so the full demo runs in ~45 s for a
readable but compact animated GIF.  Delegates all content to video_demo_v3.

Run via VHS:
    vhs demos/demo_v3.tape
"""

import time as _time_module

_orig_sleep = _time_module.sleep


# Cap every sleep to 0.08 s — keeps lines readable on screen but stays brief
def _fast_sleep(seconds: float) -> None:
    _orig_sleep(min(float(seconds), 0.08))


_time_module.sleep = _fast_sleep

# After patching, import and run the actual demo.
# Use importlib so this works whether invoked as a script or via -m.
import importlib.util as _ilu
import os as _os

_here = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("video_demo_v3", _os.path.join(_here, "video_demo_v3.py"))
_demo = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_demo)

if __name__ == "__main__":
    _demo.main()
