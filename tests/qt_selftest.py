#!/usr/bin/env python3
"""Offscreen checks for Music3 Studio (no MLX, no network). `--probe` instead shows the
window on a real display for 12 s and asserts the process is still alive — offscreen runs
cannot catch the macOS accessibility-bridge crash."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fake_model_dir(tmp: Path) -> Path:
    """A manifest plus sparse files of the right (tiny) sizes: engine.check needs only sizes."""
    files = [
        {"path": "text_encoders/te.safetensors", "sizeBytes": 300, "sha256": "x"},
        {"path": "diffusion_models/dit.safetensors", "sizeBytes": 200, "sha256": "x"},
        {"path": "vae/dav.safetensors", "sizeBytes": 100, "sha256": "x"},
        {"path": "examples/e.wav", "sizeBytes": 5, "sha256": "x"},
    ]
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "model_manifest.json").write_text(json.dumps({"files": files}))
    for f in files[:3]:
        p = tmp / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\0" * f["sizeBytes"])
    return tmp


def fake_track(folder: Path, name: str) -> Path:
    path = folder / f"{name}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\0" * 44100 * 4)
    path.with_suffix(".json").write_text(json.dumps({
        "prompt": "test brief", "lyrics": "[Instrumental]", "seconds": 12, "steps": 3,
        "seed": 42, "durationSeconds": 1.0, "elapsedSeconds": 2.0}))
    return path


def offscreen() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication
    import config
    import engine
    from qt.window import MainWindow, slug, unique_path

    app = QApplication.instance() or QApplication([])
    n = 0

    def ok(cond, what):
        nonlocal n
        n += 1
        if not cond:
            print(f"FAIL {n}: {what}")
            raise SystemExit(1)
        print(f"ok {n}: {what}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = fake_model_dir(tmp / "model")
        out = tmp / "out"
        cfg = dict(config.DEFAULTS, model_dir=str(model), output_dir=str(out),
                   engine_python=sys.executable, prompt="", seconds=12, steps=3, random_seed=False, seed=5)
        win = MainWindow(persist=False, cfg=cfg)

        # engine.check on the fake tree
        st = engine.check(model)
        ok(st["missing"] == [] and st["have"] == st["expected"] == 600, "engine.check counts weights only")
        ok(win.banner.isHidden() and win.generate_btn.isEnabled(), "weights present → banner hidden, Generate on")

        # args + output naming
        win.prompt.setPlainText("Warm acoustic folk")
        win.title.setText("My Folk Song!")
        path = win.output_path()
        ok(path.name == "my-folk-song-s5.wav", f"output name {path.name}")
        args = win.engine_args(path)
        ok(args[0].endswith("engine.py") and "--seconds" in args and args[args.index("--seed") + 1] == "5"
           and args[args.index("--lyrics") + 1] == "[Instrumental]", "engine args built")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        ok(unique_path(out, "my-folk-song-s5").name == "my-folk-song-s5-2.wav", "unique_path avoids clobber")
        path.unlink()
        ok(slug("  Ünïcode & Spaces ") == "n-code-spaces", "slug")

        # dry-run through the real engine CLI (pure python path)
        r = subprocess.run([sys.executable] + args + ["--dry-run"], capture_output=True, text=True, timeout=60)
        last = json.loads(r.stdout.strip().splitlines()[-1])
        ok(r.returncode == 0 and last.get("done") == "dry-run", f"engine --dry-run: {last}")
        r = subprocess.run([sys.executable, str(ROOT / "engine.py"), "--model-dir", str(model),
                            "--prompt", "x", "--output", str(out / "x.wav"), "--seconds", "3"],
                           capture_output=True, text=True, timeout=60)
        ok(r.returncode == 1 and "error" in json.loads(r.stdout.strip().splitlines()[-1]), "engine rejects seconds<10")

        # progress events
        win._started = time.time()
        win._on_line(json.dumps({"stage": 2, "of": 4, "progress": 0.5, "msg": "Refining step 2/4…"}))
        ok(win.progress.value() == 375 and "2/4" in win.stage_lbl.text(), "stage event → progress 37.5%")
        win._on_line("plain stderr-ish text")
        ok("plain stderr-ish text" in win.log.toPlainText(), "non-JSON lines go to the log")
        win._on_line(json.dumps({"done": str(out / "a.wav"), "elapsed": 3.0, "peak_gb": 1.0, "duration": 12.0}))
        ok(win._result and win._result["done"].endswith("a.wav"), "done event captured")

        # library + sidecar
        track = fake_track(out, "demo-s42")
        win.refresh_library(select=track)
        ok(win.tracks.count() == 1 and win.tracks.currentItem().text() == "demo-s42", "library lists the track")
        ok("test brief" in win.details.text() and "seed 42" in win.details.text(), "sidecar details shown")
        win.reuse_brief()
        ok(win.prompt.toPlainText() == "test brief" and win.seed.value() == 42 and win.instrumental.isChecked(),
           "Use as Brief restores parameters")
        sig = win._lib_sig
        win.refresh_library()
        ok(win._lib_sig == sig and win.tracks.count() == 1, "library refresh is a no-op without changes")

        # missing weights → banner + Generate disabled
        (model / "vae/dav.safetensors").write_bytes(b"\0" * 10)
        win.refresh_weights()
        ok(not win.banner.isHidden() and not win.generate_btn.isEnabled() and "0.0 GB" in win.banner.text(),
           "incomplete weights → banner, Generate off")
        (model / "model_manifest.json").unlink()
        win.refresh_weights()
        ok("not found" in win.banner.text(), "missing manifest → helpful banner")

        # theme switch does not throw
        win.set_theme("dark")
        ok(win.theme == "dark", "dark theme applied")
        win.set_theme("system")
        ok(win.cfg["theme"] == "", "system theme stored as empty")

        # stop with nothing running is safe; a QProcess object is never left dangling
        win.stop()
        ok(win.proc is None or win.proc.state() == QProcess.NotRunning, "stop() idle is safe")
        win.close()
    print(f"selftest: {n}/{n} passed")
    return 0


def probe() -> int:
    """Show the real window on screen for 12 s in a subprocess; alive == pass."""
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from PySide6.QtCore import QTimer\n"
            "from PySide6.QtWidgets import QApplication\n"
            "import config\n"
            "from qt.window import MainWindow\n"
            "app = QApplication([]); app.setStyle('Fusion')\n"
            "w = MainWindow(persist=False); w.show(); w.raise_()\n"
            "snap = sys.argv[1] if len(sys.argv) > 1 else ''\n"
            "QTimer.singleShot(9000, lambda: snap and w.grab().save(snap))\n"
            "QTimer.singleShot(12000, app.quit)\n"
            "sys.exit(app.exec())\n") % str(ROOT)
    snap = os.environ.get("M3_SNAPSHOT", "")
    p = subprocess.Popen([sys.executable, "-c", code, snap], env=env)
    time.sleep(11)
    alive = p.poll() is None
    p.wait(timeout=10)
    print("probe: window alive 11 s" if alive else f"probe: DIED early, exit {p.returncode}")
    return 0 if alive else 1


def main(argv=None) -> int:
    argv = argv or sys.argv
    return probe() if "--probe" in argv else offscreen()


if __name__ == "__main__":
    raise SystemExit(main())
