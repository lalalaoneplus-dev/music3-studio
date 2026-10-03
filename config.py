"""Music3 Studio settings — one JSON file next to the code (machine paths + UI state)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
APP_NAME = "Music3 Studio"
VERSION = "1.0"

DEFAULTS = {
    "model_dir": str(Path.home() / "models" / "MiniMax-Music3-MLX"),
    "engine_python": str(Path.home() / "models" / "music3-venv" / "bin" / "python"),
    "output_dir": str(Path.home() / "Music" / APP_NAME),
    "theme": "",            # "", "light" or "dark"; "" follows the system
    "geometry": "",
    "sash": 320,
    "mp3": True,
    "seconds": 30,
    "steps": 15,
    "seed": 7,
    "random_seed": True,
    "instrumental": True,
    "prompt": "",
    "lyrics": "",
}

PRESETS = {
    "Lo-fi study beat": "Lo-fi hip hop, mellow Rhodes piano, vinyl crackle, laid-back drums, warm bass, 80 BPM, instrumental",
    "Rock and roll": "High-energy rock and roll, gritty male vocal, crunchy guitars, boogie piano, live drums, punchy bass, 148 BPM",
    "Cinematic trailer": "Cinematic orchestral trailer, low strings, brass swells, epic percussion, building to a huge climax, instrumental",
    "Acoustic folk": "Warm acoustic folk, fingerpicked guitar, soft female vocal, intimate, gentle harmonies, 95 BPM",
    "Synth-pop": "Upbeat synth-pop, punchy bass, bright arpeggiated synths, catchy chorus, female vocal, 120 BPM",
    "Ambient": "Ambient electronic, soft evolving pads, slow textures, distant piano, no drums, instrumental",
    "Indian fusion": "Indian classical fusion, sitar and tabla with warm electronic bass, meditative groove, 90 BPM, instrumental",
}


def load(path: Path = CONFIG_PATH) -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(path.read_text()))
    except (OSError, ValueError):
        pass
    return cfg


def save(cfg: dict, path: Path = CONFIG_PATH) -> None:
    path.write_text(json.dumps({k: cfg[k] for k in DEFAULTS if k in cfg}, indent=2))
