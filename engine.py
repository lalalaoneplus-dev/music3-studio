#!/usr/bin/env python3
"""Music3 Studio engine — staged, low-memory MiniMax-Music3 generation on MLX.

Runs inside the MLX venv as a subprocess of the Qt app (or standalone). The port's own
pipeline keeps all three models resident (~12 GB); this driver loads and frees them one at a
time — AR (9.2 GB) → flow DiT (2.5 GB) → DAV decoder (0.2 GB) — so a 24 GB Mac has room for
the KV cache and activations. Every event is one JSON line on stdout:

    {"stage": 2, "of": 4, "msg": "...", "progress": 0.4}
    {"done": "/path/song.wav", "elapsed": 312.5, "peak_gb": 14.1}
    {"error": "..."}

    python engine.py --model-dir D --prompt P (--lyrics L | --lyrics-file F) [--seconds 60]
                     [--steps 30] [--seed 7] [--mp3] --output song.wav
    python engine.py --model-dir D --check      # weights present with the manifest's sizes
    python engine.py --model-dir D --verify     # sha256 every weight against the manifest
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

STAGES = 4
WEIGHT_PREFIXES = ("text_encoders/", "diffusion_models/", "vae/")


def emit(**event) -> None:
    print(json.dumps(event), flush=True)


def weight_files(model_dir: Path) -> list[dict]:
    files = json.loads((model_dir / "model_manifest.json").read_text())["files"]
    return [f for f in files if f["path"].startswith(WEIGHT_PREFIXES)]


def check(model_dir: Path) -> dict:
    """Bytes present vs expected, no MLX import. `missing` lists incomplete paths."""
    have = expected = 0
    missing = []
    for f in weight_files(model_dir):
        p = model_dir / f["path"]
        size = p.stat().st_size if p.exists() else 0
        have += min(size, f["sizeBytes"])
        expected += f["sizeBytes"]
        if size != f["sizeBytes"]:
            missing.append(f["path"])
    return {"have": have, "expected": expected, "missing": missing}


def verify(model_dir: Path) -> list[str]:
    bad = []
    for f in weight_files(model_dir):
        digest = hashlib.sha256()
        with (model_dir / f["path"]).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                digest.update(chunk)
        ok = digest.hexdigest() == f["sha256"]
        emit(verify=f["path"], ok=ok)
        if not ok:
            bad.append(f["path"])
    return bad


def write_wav(path: Path, audio, sample_rate: int) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(audio.T, -1.0, 1.0) * 32_767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())


def generate(a: argparse.Namespace) -> None:
    sys.path.insert(0, str(a.model_dir))
    import mlx.core as mx
    import numpy as np
    import minimax_mlx_model as mm

    try:  # let Metal pin up to Apple's recommended working set instead of the default cap
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    except Exception:  # noqa: BLE001
        pass

    def release() -> None:
        gc.collect()
        mx.clear_cache()

    t0 = time.time()
    frames = min(mm.MAX_AUDIO_FRAMES, max(1, round(a.seconds * mm.AUDIO_FRAMES_PER_SECOND)))
    model_dir = a.model_dir

    # Progress for the two long stages: the AR loop samples 8 codes per frame through
    # _sample_top_k (a module global), the DiT calls velocity() once per step.
    count = {"n": 0}
    original_sample = mm._sample_top_k

    def counted_sample(*args, **kw):
        count["n"] += 1
        if count["n"] % (8 * 25) == 0:
            done = count["n"] // 8
            emit(stage=1, of=STAGES, progress=min(1.0, done / frames),
                 msg=f"Composing {done}/{frames} frames ({done / mm.AUDIO_FRAMES_PER_SECOND:.0f} s of audio)…")
        return original_sample(*args, **kw)

    mm._sample_top_k = counted_sample

    emit(stage=1, of=STAGES, progress=0.0, msg="Loading the 8B composer (9.2 GB)…")
    ar = mm.Music3AR(model_dir / "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors")
    emit(stage=1, of=STAGES, progress=0.0, msg=f"Composing {frames} frames ({a.seconds:g} s)…")
    context = ar.generate(a.prompt, a.lyrics, a.seed, frames)
    n_frames = int(context.shape[1])
    del ar
    release()
    emit(stage=1, of=STAGES, progress=1.0, msg=f"Composed {n_frames} frames in {time.time() - t0:.0f} s")

    emit(stage=2, of=STAGES, progress=0.0, msg="Loading the flow transformer (2.5 GB)…")
    dit = mm.Music3DiT(model_dir / "diffusion_models/minimax_music3_dit_int8_convrot.safetensors")
    step = {"n": 0}
    original_velocity = dit.velocity

    def counted_velocity(*args, **kw):
        step["n"] += 1
        emit(stage=2, of=STAGES, progress=min(1.0, step["n"] / a.steps),
             msg=f"Refining step {step['n']}/{a.steps}…")
        return original_velocity(*args, **kw)

    dit.velocity = counted_velocity
    latent = dit.sample(context, a.steps, a.seed)
    mx.eval(latent)
    del dit, context
    release()

    emit(stage=3, of=STAGES, progress=0.0, msg="Decoding stereo audio…")
    dav = mm.Music3DAV(model_dir / "vae/minimax_music3_dav.safetensors")
    waveform = dav.decode(latent)
    mx.eval(waveform)
    samples = np.asarray(waveform, dtype=np.float32)[0]
    del dav, latent, waveform
    release()

    # Same acceptance checks as the port's pipeline.generate().
    samples = samples[:, : round(n_frames / mm.AUDIO_FRAMES_PER_SECOND * mm.SAMPLE_RATE)]
    if not np.isfinite(samples).all():
        raise RuntimeError("MiniMax-Music3 produced non-finite audio")
    peak = float(np.max(np.abs(samples)))
    if float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) <= 1e-7:
        raise RuntimeError("MiniMax-Music3 produced silent audio")
    if peak > 8.0:
        raise RuntimeError(f"MiniMax-Music3 produced an implausible peak ({peak:.3f})")
    collapse = mm.stereo_collapse_fraction(samples)
    if collapse >= 0.75:
        raise RuntimeError(f"DAV decoder collapsed a stereo channel in {collapse:.0%} of the audio")
    if peak > 0.99:
        samples *= 0.99 / peak

    emit(stage=4, of=STAGES, progress=0.0, msg=f"Writing {a.output.name}…")
    write_wav(a.output, samples, mm.SAMPLE_RATE)
    elapsed = time.time() - t0
    sidecar = {
        "prompt": a.prompt, "lyrics": a.lyrics, "seconds": a.seconds, "steps": a.steps,
        "seed": a.seed, "frames": n_frames, "durationSeconds": samples.shape[1] / mm.SAMPLE_RATE,
        "elapsedSeconds": round(elapsed, 1), "peakMemoryGB": round(mx.get_peak_memory() / 1e9, 2),
        "engine": "music3-studio staged", "model": "PocketAiHub/MiniMax-Music3-MLX",
    }
    a.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    if a.mp3 and shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(a.output),
                        "-codec:a", "libmp3lame", "-q:a", "2", str(a.output.with_suffix(".mp3"))],
                       check=False)
    emit(done=str(a.output), elapsed=round(elapsed, 1), peak_gb=sidecar["peakMemoryGB"],
         duration=round(sidecar["durationSeconds"], 2))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--check", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="validate arguments and weights only")
    p.add_argument("--prompt")
    lyrics = p.add_mutually_exclusive_group()
    lyrics.add_argument("--lyrics")
    lyrics.add_argument("--lyrics-file", type=Path)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mp3", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)

    try:
        if a.check:
            emit(**check(a.model_dir))
            return 0
        if a.verify:
            bad = verify(a.model_dir)
            emit(done="verify", bad=bad)
            return 1 if bad else 0
        if not a.prompt or not a.output:
            raise SystemExit("--prompt and --output are required for generation")
        a.lyrics = (a.lyrics_file.read_text(encoding="utf-8") if a.lyrics_file else a.lyrics) or "[Instrumental]"
        a.lyrics = a.lyrics.strip()
        a.prompt = a.prompt.strip()
        if not 10 <= a.seconds <= 300:
            raise ValueError("seconds must be between 10 and 300")
        if not 1 <= a.steps <= 30:
            raise ValueError("steps must be between 1 and 30")
        status = check(a.model_dir)
        if status["missing"]:
            raise FileNotFoundError("weights incomplete: " + ", ".join(status["missing"]))
        if a.dry_run:
            emit(done="dry-run", output=str(a.output))
            return 0
        generate(a)
        return 0
    except Exception as exc:  # noqa: BLE001
        emit(error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
