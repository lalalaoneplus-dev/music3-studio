"""Music3 Studio main window — library + player on the left, the brief on the right.

Generation runs in the MLX venv as a QProcess (engine.py) so the window never blocks and a
Metal out-of-memory can only kill the job, not the app. Widgets are only touched from the
GUI thread; QProcess signals already arrive there.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import time
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPalette
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter, QVBoxLayout,
    QWidget,
)

import config
import engine

ROOT = config.ROOT
PALETTE = {
    "light": {"bg": "#f4f4f6", "panel": "#ffffff", "field": "#ffffff", "text": "#1d1d1f",
              "muted": "#63636b", "accent": "#5b3fd6", "accent_fg": "#ffffff",
              "ok": "#1a7f37", "warn": "#8a5a00", "fail": "#b42318",
              "sel": "#e4dcff", "line": "#d6d6db"},
    "dark":  {"bg": "#1b1b1d", "panel": "#26262a", "field": "#1f1f22", "text": "#f0f0f4",
              "muted": "#a2a2aa", "accent": "#a78bfa", "accent_fg": "#10161f",
              "ok": "#57c777", "warn": "#e0b341", "fail": "#ff7a6e",
              "sel": "#3b2f6e", "line": "#3c3c42"},
}
ABOUT = (f"{config.APP_NAME} {config.VERSION}\n\n"
         "Local text-to-music on Apple Silicon. Runs the MiniMax-Music3 open-weights model "
         "through the community MLX port, one stage at a time so it fits a 24 GB Mac.\n\n"
         "Weights: PocketAiHub/MiniMax-Music3-MLX (MiniMax-Music3 Community License).")


def system_dark() -> bool:
    try:
        return QApplication.instance().styleHints().colorScheme() == Qt.ColorScheme.Dark
    except Exception:  # noqa: BLE001
        return False


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:40] or "track"


def unique_path(folder: Path, stem: str, suffix: str = ".wav") -> Path:
    path = folder / f"{stem}{suffix}"
    n = 2
    while path.exists():
        path = folder / f"{stem}-{n}{suffix}"
        n += 1
    return path


def fmt_gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def mmss(ms: int) -> str:
    s = max(0, ms // 1000)
    return f"{s // 60}:{s % 60:02d}"


class MainWindow(QMainWindow):
    def __init__(self, persist: bool = True, cfg: dict | None = None):
        super().__init__()
        self.persist = persist
        self.cfg = cfg if cfg is not None else config.load()
        self.proc: QProcess | None = None
        self.fetch: QProcess | None = None
        self._lib_sig: tuple = ()
        self._started = 0.0
        self.setWindowTitle(config.APP_NAME)
        self.theme = self.cfg.get("theme") or ("dark" if system_dark() else "light")
        self.pal = PALETTE[self.theme]
        self._icon()
        self._menu()

        self.split = QSplitter(Qt.Horizontal)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self._sidebar())
        self.split.addWidget(self._work())
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        sash = int(self.cfg.get("sash") or 320)
        self.split.setSizes([sash, max(600, 1200 - sash)])
        self.setCentralWidget(self.split)
        self.setMinimumSize(1000, 680)
        self.statusBar().showMessage("Ready")

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(lambda d: self.seek.setRange(0, int(d)))
        self.player.playbackStateChanged.connect(self._on_playstate)

        self._load_form()
        self.apply_theme()
        geo = re.match(r"(\d+)x(\d+)(?:\+(-?\d+)\+(-?\d+))?", self.cfg.get("geometry") or "")
        self.resize(int(geo.group(1)), int(geo.group(2))) if geo else self.resize(1280, 820)
        if geo and geo.group(3):
            self.move(int(geo.group(3)), int(geo.group(4)))

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_weights)
        self.timer.timeout.connect(self.refresh_library)
        self.timer.start(3000)
        self.refresh_weights()
        self.refresh_library()

    # ---------------------------------------------------------------- chrome
    def _icon(self):
        png = ROOT / "packaging" / "Music3Studio_256.png"
        if png.exists():
            self.setWindowIcon(QIcon(str(png)))

    def _menu(self):
        bar = self.menuBar()
        f = bar.addMenu("File")
        f.addAction(self._act("Open Output Folder", lambda: self._open(self.output_dir()), "Ctrl+O"))
        f.addAction(self._act("Choose Output Folder…", self.choose_output))
        f.addAction(self._act("Open Model Folder", lambda: self._open(self.model_dir())))
        f.addSeparator()
        f.addAction(self._act("Quit", self.close, "Ctrl+Q"))
        v = bar.addMenu("View")
        for name in ("System", "Light", "Dark"):
            v.addAction(self._act(name, lambda _=False, n=name: self.set_theme(n.lower())))
        m = bar.addMenu("Model")
        m.addAction(self._act("Download Weights", self.download_weights))
        m.addAction(self._act("Verify Checksums", self.verify_weights))
        m.addAction(self._act("Choose Model Folder…", self.choose_model))
        h = bar.addMenu("Help")
        h.addAction(self._act("About", lambda: QMessageBox.information(self, "About", ABOUT)))

    def _act(self, name, slot, key=None) -> QAction:
        a = QAction(name, self)
        a.triggered.connect(slot)
        if key:
            a.setShortcut(QKeySequence(key))
        return a

    def _sidebar(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(260)
        lay = QVBoxLayout(w)
        title = QLabel("Library")
        title.setStyleSheet("font-weight: 600; font-size: 15px;")
        lay.addWidget(title)
        self.tracks = QListWidget()
        self.tracks.currentItemChanged.connect(self._on_track)
        lay.addWidget(self.tracks, 1)
        self.details = QLabel("No tracks yet. Write a brief and press Generate.")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.details)
        row = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        self.time_lbl = QLabel("0:00 / 0:00")
        row.addWidget(self.play_btn)
        row.addWidget(self.time_lbl, 1)
        lay.addLayout(row)
        self.seek = QSlider(Qt.Horizontal)
        self.seek.sliderMoved.connect(self.player_seek)
        lay.addWidget(self.seek)
        row2 = QHBoxLayout()
        self.reveal_btn = QPushButton("Reveal in Finder")
        self.reveal_btn.clicked.connect(self.reveal)
        self.reuse_btn = QPushButton("Use as Brief")
        self.reuse_btn.clicked.connect(self.reuse_brief)
        for b in (self.reveal_btn, self.reuse_btn):
            b.setEnabled(False)
            row2.addWidget(b)
        lay.addLayout(row2)
        return w

    def _work(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setObjectName("banner")
        self.banner.hide()
        lay.addWidget(self.banner)

        brief = QGroupBox("Brief")
        bl = QVBoxLayout(brief)
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset"))
        self.preset = QComboBox()
        self.preset.addItem("— pick a starting point —")
        self.preset.addItems(list(config.PRESETS))
        self.preset.currentTextChanged.connect(self._on_preset)
        row.addWidget(self.preset, 1)
        bl.addLayout(row)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("Genre, mood, instruments, vocal, tempo… e.g. "
                                       "\"Warm acoustic folk, fingerpicked guitar, soft vocal, 95 BPM\"")
        self.prompt.setMaximumHeight(90)
        bl.addWidget(self.prompt)
        lay.addWidget(brief)

        lyr = QGroupBox("Lyrics")
        ll = QVBoxLayout(lyr)
        self.instrumental = QCheckBox("Instrumental (no lyrics)")
        self.instrumental.toggled.connect(lambda on: self.lyrics.setEnabled(not on))
        ll.addWidget(self.instrumental)
        self.lyrics = QPlainTextEdit()
        self.lyrics.setPlaceholderText("[Verse]\nNeon on the dashboard, midnight in the street\n\n[Chorus]\nTurn it up, let the good times roll")
        ll.addWidget(self.lyrics)
        lay.addWidget(lyr, 1)

        render = QGroupBox("Render")
        form = QFormLayout(render)
        self.title = QLineEdit()
        self.title.setPlaceholderText("file name (optional)")
        form.addRow("Title", self.title)
        self.seconds = QSpinBox()
        self.seconds.setRange(10, 300)
        self.seconds.setSuffix(" s")
        self.steps = QSpinBox()
        self.steps.setRange(1, 30)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.random_seed = QCheckBox("Random")
        self.random_seed.toggled.connect(lambda on: self.seed.setEnabled(not on))
        self.mp3 = QCheckBox("Also write MP3")
        r = QHBoxLayout()
        r.addWidget(self.seconds)
        r.addWidget(QLabel("Steps"))
        r.addWidget(self.steps)
        r.addWidget(QLabel("Seed"))
        r.addWidget(self.seed)
        r.addWidget(self.random_seed)
        r.addWidget(self.mp3)
        r.addStretch(1)
        form.addRow("Length", r)
        lay.addWidget(render)

        btns = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setDefault(True)
        self.generate_btn.clicked.connect(self.generate)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        btns.addWidget(self.generate_btn)
        btns.addWidget(self.stop_btn)
        self.stage_lbl = QLabel("")
        btns.addWidget(self.stage_lbl, 1)
        lay.addLayout(btns)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        lay.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMaximumHeight(150)
        lay.addWidget(self.log)
        return w

    # ---------------------------------------------------------------- state
    def model_dir(self) -> Path:
        return Path(self.cfg["model_dir"]).expanduser()

    def output_dir(self) -> Path:
        p = Path(self.cfg["output_dir"]).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _load_form(self):
        c = self.cfg
        self.prompt.setPlainText(c.get("prompt") or "")
        self.lyrics.setPlainText(c.get("lyrics") or "")
        self.instrumental.setChecked(bool(c.get("instrumental", True)))
        self.lyrics.setEnabled(not self.instrumental.isChecked())
        self.seconds.setValue(int(c.get("seconds", 30)))
        self.steps.setValue(int(c.get("steps", 15)))
        self.seed.setValue(int(c.get("seed", 7)))
        self.random_seed.setChecked(bool(c.get("random_seed", True)))
        self.seed.setEnabled(not self.random_seed.isChecked())
        self.mp3.setChecked(bool(c.get("mp3", True)))

    def _save(self):
        if not self.persist:
            return
        c = self.cfg
        c.update(prompt=self.prompt.toPlainText(), lyrics=self.lyrics.toPlainText(),
                 instrumental=self.instrumental.isChecked(), seconds=self.seconds.value(),
                 steps=self.steps.value(), seed=self.seed.value(),
                 random_seed=self.random_seed.isChecked(), mp3=self.mp3.isChecked(),
                 sash=self.split.sizes()[0], theme=self.cfg.get("theme") or "",
                 geometry=f"{self.width()}x{self.height()}+{self.x()}+{self.y()}")
        config.save(c)

    def closeEvent(self, e):
        self._save()
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
        super().closeEvent(e)

    def set_theme(self, name: str):
        self.cfg["theme"] = "" if name == "system" else name
        self.theme = self.cfg["theme"] or ("dark" if system_dark() else "light")
        self.apply_theme()
        self._save()

    def apply_theme(self):
        p = self.pal = PALETTE[self.theme]
        qp = QPalette()
        for role, key in ((QPalette.Window, "bg"), (QPalette.WindowText, "text"),
                          (QPalette.Base, "field"), (QPalette.AlternateBase, "panel"),
                          (QPalette.Text, "text"), (QPalette.Button, "panel"),
                          (QPalette.ButtonText, "text"), (QPalette.Highlight, "sel"),
                          (QPalette.HighlightedText, "text"), (QPalette.PlaceholderText, "muted"),
                          (QPalette.Link, "accent")):
            qp.setColor(role, QColor(p[key]))
        app = QApplication.instance()
        app.setPalette(qp)
        app.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {p['bg']}; color: {p['text']}; }}
            QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QListWidget {{
                background: {p['field']}; color: {p['text']};
                border: 1px solid {p['line']}; padding: 4px; }}
            QGroupBox {{ background: {p['panel']}; border: 1px solid {p['line']};
                         margin-top: 10px; padding: 8px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
            QPushButton {{ background: {p['panel']}; color: {p['text']};
                           border: 1px solid {p['line']}; padding: 6px 14px; }}
            QPushButton:default {{ background: {p['accent']}; color: {p['accent_fg']}; }}
            QPushButton:disabled {{ background: {p['panel']}; color: {p['muted']}; }}
            QProgressBar {{ border: 1px solid {p['line']}; background: {p['field']}; height: 8px; }}
            QProgressBar::chunk {{ background: {p['accent']}; }}
            QLabel#banner {{ background: {p['panel']}; border: 1px solid {p['warn']};
                             color: {p['warn']}; padding: 8px; }}
        """)

    # ---------------------------------------------------------------- weights
    def weights_status(self) -> dict:
        try:
            return engine.check(self.model_dir())
        except (OSError, ValueError, KeyError) as exc:
            return {"have": 0, "expected": 0, "missing": ["manifest"], "error": str(exc)}

    def refresh_weights(self):
        st = self.weights_status()
        busy = self.proc is not None and self.proc.state() != QProcess.NotRunning
        if st.get("error"):
            self.banner.setText(f"Model folder not found or incomplete: {self.model_dir()}\n"
                                "Model ▸ Download Weights fetches the 11.9 GB MLX port, or pick the folder under Model ▸ Choose Model Folder.")
            self.banner.show()
        elif st["missing"]:
            fetching = self.fetch is not None and self.fetch.state() != QProcess.NotRunning
            lock = (self.model_dir() / ".fetch.lock").exists()
            verb = "Downloading" if (fetching or lock) else "Weights incomplete"
            self.banner.setText(f"{verb}: {fmt_gb(st['have'])} of {fmt_gb(st['expected'])}"
                                + ("" if (fetching or lock) else " — Model ▸ Download Weights to resume."))
            self.banner.show()
        else:
            self.banner.hide()
        ready = not st["missing"] and not busy
        self.generate_btn.setEnabled(ready)
        if not busy:
            self.statusBar().showMessage("Weights ready" if not st["missing"] else "Weights incomplete")

    def download_weights(self):
        if self.fetch and self.fetch.state() != QProcess.NotRunning:
            self.log_line("Download already running.")
            return
        self.fetch = QProcess(self)
        self.fetch.setProcessChannelMode(QProcess.MergedChannels)
        self.fetch.readyReadStandardOutput.connect(
            lambda: self.log_line(bytes(self.fetch.readAllStandardOutput()).decode(errors="replace").rstrip()))
        self.fetch.finished.connect(lambda code, _s: self.log_line(f"Download finished (exit {code})."))
        self.fetch.start("/bin/bash", [str(ROOT / "packaging" / "fetch-weights.sh"), str(self.model_dir())])
        self.log_line(f"Fetching weights into {self.model_dir()}…")

    def verify_weights(self):
        self.log_line("Verifying checksums (about a minute)…")
        self._run_engine(["--verify"], on_done=lambda ok: self.log_line("Checksums OK" if ok else "CHECKSUM MISMATCH — re-download"))

    def choose_model(self):
        d = QFileDialog.getExistingDirectory(self, "MiniMax-Music3-MLX folder", str(self.model_dir()))
        if d:
            self.cfg["model_dir"] = d
            self._save()
            self.refresh_weights()

    def choose_output(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", str(self.output_dir()))
        if d:
            self.cfg["output_dir"] = d
            self._save()
            self.refresh_library()

    # ---------------------------------------------------------------- generate
    def _on_preset(self, name: str):
        text = config.PRESETS.get(name)
        if text:
            self.prompt.setPlainText(text)
            if not self.title.text():
                self.title.setText(name)

    def engine_args(self, output: Path) -> list[str]:
        seed = random.randrange(2**31) if self.random_seed.isChecked() else self.seed.value()
        self.seed.setValue(seed)
        lyrics = "[Instrumental]" if self.instrumental.isChecked() else self.lyrics.toPlainText().strip() or "[Instrumental]"
        args = [str(ROOT / "engine.py"), "--model-dir", str(self.model_dir()),
                "--prompt", self.prompt.toPlainText().strip(), "--lyrics", lyrics,
                "--seconds", str(self.seconds.value()), "--steps", str(self.steps.value()),
                "--seed", str(seed), "--output", str(output)]
        if self.mp3.isChecked():
            args.append("--mp3")
        return args

    def output_path(self) -> Path:
        stem = slug(self.title.text() or self.prompt.toPlainText()[:40])
        return unique_path(self.output_dir(), f"{stem}-s{self.seed.value()}")

    def generate(self):
        if not self.prompt.toPlainText().strip():
            QMessageBox.warning(self, config.APP_NAME, "Write a brief first: genre, mood, instruments, tempo.")
            return
        if self.proc and self.proc.state() != QProcess.NotRunning:
            return
        self._save()
        out = self.output_path()
        args = self.engine_args(out)
        out = Path(args[args.index("--output") + 1])
        self.log.clear()
        self.log_line(f"→ {out.name}  ({self.seconds.value()} s, {self.steps.value()} steps, seed {self.seed.value()})")
        self.progress.setValue(0)
        self._started = time.time()
        self._run_engine(args[1:], on_done=self._on_generated, output=out)

    def _run_engine(self, args: list[str], on_done, output: Path | None = None):
        self.proc = QProcess(self)
        self.proc.setProgram(self.cfg["engine_python"])
        self.proc.setArguments([str(ROOT / "engine.py"), "--model-dir", str(self.model_dir())] + args
                               if args and args[0].startswith("--") and "--model-dir" not in args else args)
        self._buf = b""
        self._result = None
        self.proc.readyReadStandardOutput.connect(self._read_stdout)
        self.proc.readyReadStandardError.connect(
            lambda: self.log_line(bytes(self.proc.readAllStandardError()).decode(errors="replace").rstrip()))
        self.proc.finished.connect(lambda code, _s: self._finished(code, on_done, output))
        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("Working…")
        self.proc.start()

    def _read_stdout(self):
        self._buf += bytes(self.proc.readAllStandardOutput())
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._on_line(line.decode(errors="replace"))

    def _on_line(self, line: str):
        line = line.strip()
        if not line:
            return
        try:
            ev = json.loads(line)
        except ValueError:
            self.log_line(line)
            return
        if "stage" in ev:
            stage, of = ev["stage"], ev.get("of", engine.STAGES)
            frac = ((stage - 1) + float(ev.get("progress", 0.0))) / of
            self.progress.setValue(int(frac * 1000))
            elapsed = time.time() - self._started
            self.stage_lbl.setText(f"{stage}/{of} · {ev['msg']}  ·  {elapsed:.0f} s")
            if ev.get("progress", 0.0) in (0.0, 1.0):
                self.log_line(ev["msg"])
        elif "verify" in ev:
            self.log_line(f"{'✓' if ev['ok'] else '✗'} {ev['verify']}")
        elif "error" in ev:
            self.log_line("ERROR " + ev["error"])
            self._result = ev
        elif "done" in ev:
            self._result = ev
        else:
            self.log_line(line)

    def _finished(self, code: int, on_done, output: Path | None):
        self.stop_btn.setEnabled(False)
        self.proc = None
        ok = code == 0 and self._result is not None and "error" not in self._result
        if not ok and self._result is None:
            self.log_line(f"Engine stopped (exit {code}).")
        self.stage_lbl.setText("")
        on_done(ok and self._result)
        self.refresh_weights()

    def _on_generated(self, result):
        if not result:
            self.statusBar().showMessage("Generation failed — see log")
            self.progress.setValue(0)
            return
        self.progress.setValue(1000)
        self.log_line(f"Done in {result['elapsed']:.0f} s · {result['duration']:.1f} s of audio · "
                      f"peak memory {result['peak_gb']:.1f} GB")
        self.statusBar().showMessage(f"Rendered {Path(result['done']).name} in {result['elapsed']:.0f} s")
        self.refresh_library(select=Path(result["done"]))

    def stop(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.log_line("Stopping…")
            self.proc.kill()

    def log_line(self, text: str):
        if text:
            self.log.appendPlainText(text)

    # ---------------------------------------------------------------- library
    def refresh_library(self, select: Path | None = None):
        try:
            files = sorted(self.output_dir().glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            files = []
        sig = tuple((p.name, p.stat().st_mtime) for p in files)
        if sig == self._lib_sig and select is None:
            return
        self._lib_sig = sig
        current = select or (self.tracks.currentItem().data(Qt.UserRole) if self.tracks.currentItem() else None)
        self.tracks.blockSignals(True)
        self.tracks.clear()
        for p in files:
            item = QListWidgetItem(p.stem)
            item.setData(Qt.UserRole, p)
            self.tracks.addItem(item)
            if current and Path(current) == p:
                self.tracks.setCurrentItem(item)
        self.tracks.blockSignals(False)
        if self.tracks.currentItem():
            self._on_track(self.tracks.currentItem(), None)

    def _on_track(self, item: QListWidgetItem | None, _prev=None):
        path = item.data(Qt.UserRole) if item else None
        for b in (self.play_btn, self.reveal_btn, self.reuse_btn):
            b.setEnabled(path is not None)
        if not path:
            return
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        meta = self.sidecar(path)
        if meta:
            self.details.setText(
                f"{meta.get('prompt', '')}\n\n{meta.get('durationSeconds', 0):.0f} s · "
                f"{meta.get('steps')} steps · seed {meta.get('seed')} · "
                f"rendered in {meta.get('elapsedSeconds', 0):.0f} s")
        else:
            self.details.setText(path.name)

    @staticmethod
    def sidecar(path: Path) -> dict:
        try:
            return json.loads(path.with_suffix(".json").read_text())
        except (OSError, ValueError):
            return {}

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_playstate(self, state):
        self.play_btn.setText("⏸ Pause" if state == QMediaPlayer.PlayingState else "▶ Play")

    def _on_position(self, pos: int):
        if not self.seek.isSliderDown():
            self.seek.setValue(int(pos))
        self.time_lbl.setText(f"{mmss(pos)} / {mmss(self.player.duration())}")

    def player_seek(self, pos: int):
        self.player.setPosition(int(pos))

    def reveal(self):
        item = self.tracks.currentItem()
        if item:
            subprocess.Popen(["open", "-R", str(item.data(Qt.UserRole))])

    def reuse_brief(self):
        item = self.tracks.currentItem()
        meta = self.sidecar(item.data(Qt.UserRole)) if item else {}
        if not meta:
            return
        self.prompt.setPlainText(meta.get("prompt", ""))
        lyr = meta.get("lyrics", "")
        self.instrumental.setChecked(lyr.strip().lower() == "[instrumental]")
        self.lyrics.setPlainText("" if self.instrumental.isChecked() else lyr)
        self.seconds.setValue(int(round(meta.get("seconds", 30))))
        self.steps.setValue(int(meta.get("steps", 15)))
        self.random_seed.setChecked(False)
        self.seed.setValue(int(meta.get("seed", 7)))

    @staticmethod
    def _open(path: Path):
        subprocess.Popen(["open", str(path)])
