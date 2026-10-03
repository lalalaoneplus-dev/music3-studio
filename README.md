# Music3 Studio

Local text-to-music on Apple Silicon: a PySide6 desktop app around the community MLX port of
MiniMax-Music3 (PocketAiHub/MiniMax-Music3-MLX, int8, 11.9 GB).

- `engine.py` — staged generator run inside the MLX venv (AR → DiT → DAV, each loaded then freed so
  a 24 GB Mac fits). JSON events on stdout. `--check` / `--verify` / `--dry-run`.
- `app_qt.py` + `qt/window.py` — the window (brief, lyrics, render controls, library + player).
- `config.json` — machine paths (model dir, venv python, output dir) + UI state; created on first save.
- `packaging/fetch-weights.sh` — resumable curl download of the weights.
- `packaging/make_app_qt.sh` — builds `~/Desktop/Music3 Studio.app` (thin launcher, code stays here).
- `tests/qt_selftest.py` — `python3 app_qt.py --selftest` (offscreen) and `--probe` (12 s on screen).

Runtime: `~/models/music3-venv` (mlx 0.30.6) + `~/models/MiniMax-Music3-MLX`.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/lalalaoneplus-dev/music3-studio/main/install.sh | bash
```

Sets up the Apple Silicon UI and MLX engine environments, then launches Music3 Studio.
