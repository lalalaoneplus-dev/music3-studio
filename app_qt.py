#!/usr/bin/env python3
"""Music3 Studio — Qt entry. `--selftest` runs the offscreen checks, `--probe` shows the
window on screen for 12 s (accessibility-bridge liveness, see tests/qt_selftest.py)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if "--selftest" in argv or "--probe" in argv:
        from tests.qt_selftest import main as run_selftest
        return run_selftest(argv)
    from PySide6.QtWidgets import QApplication
    import config
    from qt.window import MainWindow

    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(config.APP_NAME)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
