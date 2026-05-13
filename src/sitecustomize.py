"""Project runtime compatibility hooks.

This file is imported automatically by Python when `src` is on PYTHONPATH.
It keeps upstream WeKWS/Wenet code runnable on Windows without editing the
installed Python environment.
"""

import platform


if platform.system() == "Windows":
    try:
        import torchaudio.utils.sox_utils as sox_utils

        def _set_buffer_size_noop(*args, **kwargs):
            return None

        sox_utils.set_buffer_size = _set_buffer_size_noop
    except Exception:
        pass
