"""Assemble original UI captures into a silent, captioned 1080p release walkthrough.

Install the media-only helper outside the application environment:
  uv pip install --python python/.venv/bin/python --target .cache/video-tools --no-deps imageio-ffmpeg==0.6.0 Pillow==12.0.0
Run: python tools/assemble_walkthrough.py .qps/walkthroughs/v0.4/TIMESTAMP/manifest.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(".cache/video-tools").resolve()))
from PIL import Image, ImageDraw, ImageFont


def assemble(manifest: Path, seconds=240):
    if not 180 <= seconds <= 300:
        raise ValueError("The walkthrough must last between three and five minutes")
    root = manifest.resolve().parent
    value = json.loads(manifest.read_text())
    captures = value["captures"]
    if not captures:
        raise ValueError("The screenshot manifest is empty")
    sys.path.insert(0, str(Path(".cache/video-tools").resolve()))
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    fonts = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    font_path = next((p for p in fonts if p.is_file()), None)
    if font_path is None:
        raise ValueError("Install a readable TrueType font before rendering captions")
    font = ImageFont.truetype(str(font_path), 32)
    small = ImageFont.truetype(str(font_path), 21)
    frames = []
    for index, capture in enumerate(captures):
        source = (root / capture["path"]).resolve()
        if not source.is_relative_to(root) or "\n" in str(source):
            raise ValueError(
                "Capture paths must remain inside the walkthrough directory"
            )
        if hashlib.sha256(source.read_bytes()).hexdigest() != capture["sha256"]:
            raise ValueError("Original screenshot checksum mismatch")
        original = Image.open(source).convert("RGB")
        frame = Image.new("RGB", (1920, 1080), "#101626")
        # Original 1920x936 captures retain their text size; other viewports are letterboxed.
        original.thumbnail((1920, 936), Image.Resampling.LANCZOS)
        frame.paste(
            original, ((1920 - original.width) // 2, (936 - original.height) // 2)
        )
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 936, 1920, 1080), fill="#101626")
        lines = textwrap.wrap(capture["caption"], width=104)
        if len(lines) > 2:
            raise ValueError("Shorten the caption to two readable lines")
        for row, line in enumerate(lines):
            draw.text((42, 953 + row * 39), line, font=font, fill="white")
        draw.text(
            (42, 1040),
            f"QPS v0.4  ·  {capture['step']}  ·  {capture['capturedAt']}",
            font=small,
            fill="#b0bfdc",
        )
        output = source.parent / f"captioned-{index:03}.png"
        frame.save(output)
        frames.append(output)
    duration = seconds / len(frames)
    playlist = root / "frames.ffconcat"

    def entry(path):
        return "file '" + str(path).replace("\\", "\\\\").replace("'", "'\\''") + "'\n"

    playlist.write_text(
        "ffconcat version 1.0\n"
        + "".join(entry(p) + f"duration {duration:.8f}\n" for p in frames)
        + entry(frames[-1])
    )
    movie = root / "walkthrough.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-safe",
            "0",
            "-f",
            "concat",
            "-i",
            str(playlist),
            "-vf",
            f"tpad=stop_mode=clone:stop_duration={seconds},fps=30",
            "-t",
            str(seconds),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-fps_mode",
            "cfr",
            "-threads",
            "4",
            "-movflags",
            "+faststart",
            str(movie),
        ],
        check=True,
    )
    reader = imageio_ffmpeg.read_frames(str(movie))
    try:
        metadata = next(reader)
    finally:
        reader.close()
    if (
        tuple(metadata["size"]) != (1920, 1080)
        or metadata["codec"] != "h264"
        or abs(metadata["duration"] - seconds) > 1
    ):
        raise ValueError(
            "Encoded video did not meet resolution, codec or duration requirements"
        )
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(movie), "-f", "null", "-"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    evidence = {
        "path": str(movie),
        "durationSeconds": metadata["duration"],
        "width": 1920,
        "height": 1080,
        "codec": "h264",
        "audio": False,
        "originalCaptures": len(captures),
        "originalChecksumsVerified": True,
        "fullDecodeVerified": True,
        "sha256": hashlib.sha256(movie.read_bytes()).hexdigest(),
    }
    (root / "video-verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seconds", type=int, default=240)
    arguments = parser.parse_args()
    print(json.dumps(assemble(arguments.manifest, arguments.seconds), indent=2))
