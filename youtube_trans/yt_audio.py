#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube audio/video download & transcription tool.
Uses yt-dlp with cookie auth, and faster-whisper for transcription.

Usage:
  python yt_audio.py --export-cookies
  python yt_audio.py "https://www.youtube.com/watch?v=XXXX"
  python yt_audio.py --transcribe
  python yt_audio.py --transcribe ~/Music/yt_audio/file.m4a
  python yt_audio.py --transcribe --model small file.m4a
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_AUDIO_DIR = Path.home() / "Music" / "yt_audio"
DEFAULT_VIDEO_DIR = Path.home() / "Movies" / "yt_video"
DEFAULT_TRANSCRIPT_DIR = Path.home() / "Documents" / "yt_transcripts"
COOKIES_FILE = Path(__file__).parent / "cookies.txt"

AUDIO_FORMATS = ("mp3", "m4a", "flac", "wav", "opus", "aac")
VIDEO_FORMATS = ("mp4", "mkv", "webm")
MEDIA_FORMATS = AUDIO_FORMATS + VIDEO_FORMATS
WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")

# Extra yt-dlp flags applied to every network call
# --js-runtimes node: use Node.js for YouTube EJS n-challenge solving
_YT_DLP_EXTRA = ["--legacy-server-connect", "--js-runtimes", "node"]


def _yt_dlp_cmd() -> str:
    """Resolve yt-dlp executable, preferring the venv Scripts dir."""
    import shutil
    # Check env override first
    override = os.environ.get("RHINO_YT_DLP") or ""
    if override and Path(override).exists():
        return override
    # Same directory as this Python interpreter
    python_dir = Path(sys.executable).resolve().parent
    for candidate in [python_dir / "yt-dlp.exe", python_dir / "yt-dlp"]:
        if candidate.exists():
            return str(candidate)
    # Fall back to PATH
    found = shutil.which("yt-dlp")
    if found:
        return found
    return "yt-dlp"


def _proxy_args() -> list[str]:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""
    if proxy:
        return ["--proxy", proxy]
    return []


def export_cookies(browser: str = "chrome") -> None:
    print(f"-> Exporting YouTube cookies from {browser} to {COOKIES_FILE} ...")
    cmd = [
        _yt_dlp_cmd(),
        *_YT_DLP_EXTRA,
        *_proxy_args(),
        "--cookies-from-browser", browser,
        "--cookies", str(COOKIES_FILE),
        "--skip-download",
        "--no-warnings",
        "https://www.youtube.com",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"[ERR] Export failed: {result.stderr.strip()}", file=sys.stderr)
        print("  Hint: make sure the browser is closed or allows yt-dlp to read cookies",
              file=sys.stderr)
        sys.exit(1)
    print(f"[OK] Cookies exported to {COOKIES_FILE}")


def _build_cookie_args(browser: str | None, use_cookies_file: bool = True) -> list[str]:
    if browser:
        return ["--cookies-from-browser", browser]
    if use_cookies_file and COOKIES_FILE.exists():
        return ["--cookies", str(COOKIES_FILE)]
    # No cookies — fine for public videos
    return []


def _fetch_playlist_ids(url: str, browser: str | None = None) -> list[str]:
    cmd = [
        _yt_dlp_cmd(),
        *_YT_DLP_EXTRA,
        *_proxy_args(),
        "--flat-playlist",
        "--print", "%(id)s",
        "--no-warnings",
        *_build_cookie_args(browser),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"[ERR] Playlist fetch failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]


def download_audio(
    url: str,
    output_dir: Path = DEFAULT_AUDIO_DIR,
    audio_format: str = "m4a",
    browser: str | None = None,
    use_cookies_file: bool = True,
    playlist: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(title)s [%(id)s].%(ext)s")

    cmd = [
        _yt_dlp_cmd(),
        *_YT_DLP_EXTRA,
        *_proxy_args(),
        *([] if playlist else ["--no-playlist"]),
        "-x",
        "--audio-format", audio_format,
        "--audio-quality", "0",
        "-o", output_template,
        "--embed-thumbnail",
        "--embed-metadata",
        "--no-overwrites",
        "--progress",
        *_build_cookie_args(browser, use_cookies_file),
        url,
    ]

    print(f"-> Downloading audio...")
    print(f"   URL:    {url}")
    print(f"   Format: {audio_format}")
    print(f"   Dest:   {output_dir}")

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\n[OK] Download complete: {output_dir}")
    else:
        print(f"\n[ERR] Download failed (exit code: {result.returncode})", file=sys.stderr)
        sys.exit(1)


def download_video(
    url: str,
    output_dir: Path = DEFAULT_VIDEO_DIR,
    video_format: str = "mp4",
    resolution: str = "best",
    browser: str | None = None,
    use_cookies_file: bool = True,
    playlist: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(title)s [%(id)s].%(ext)s")

    if resolution == "best":
        fmt = "bestvideo+bestaudio/best"
    else:
        fmt = f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]"

    cmd = [
        _yt_dlp_cmd(),
        *_YT_DLP_EXTRA,
        *_proxy_args(),
        *([] if playlist else ["--no-playlist"]),
        "-f", fmt,
        "--merge-output-format", video_format,
        "-o", output_template,
        "--embed-thumbnail",
        "--embed-metadata",
        "--no-overwrites",
        "--progress",
        *_build_cookie_args(browser, use_cookies_file),
        url,
    ]

    print(f"-> Downloading video...")
    print(f"   URL:        {url}")
    print(f"   Format:     {video_format}")
    print(f"   Resolution: {resolution}")
    print(f"   Dest:       {output_dir}")

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\n[OK] Download complete: {output_dir}")
    else:
        print(f"\n[ERR] Download failed (exit code: {result.returncode})", file=sys.stderr)
        sys.exit(1)


def transcribe(
    input_path: Path | None = None,
    output_dir: Path = DEFAULT_TRANSCRIPT_DIR,
    model_size: str = "large-v3",
    language: str = "zh",
    device: str = "cpu",
) -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[ERR] faster-whisper not installed. Run: pip install faster-whisper", file=sys.stderr)
        sys.exit(1)

    if input_path and input_path.is_file():
        files = [input_path]
    else:
        files = []
        for d in [DEFAULT_AUDIO_DIR, DEFAULT_VIDEO_DIR]:
            if d.exists():
                files.extend(
                    f for f in sorted(d.iterdir())
                    if f.suffix.lstrip(".") in MEDIA_FORMATS
                )

    if not files:
        print("No media files found to transcribe")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for f in files:
        txt_path = output_dir / (f.stem + ".txt")
        if txt_path.exists():
            print(f"  [SKIP] Already transcribed: {f.name}")
        else:
            todo.append(f)

    if not todo:
        print(f"\nAll files already transcribed. Output: {output_dir}")
        return

    print(f"\n-> Loading Whisper model ({model_size}, device={device})...")
    compute_type = "int8" if device == "cpu" else "auto"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("[OK] Model loaded\n")

    for i, media_file in enumerate(todo, 1):
        txt_path = output_dir / (media_file.stem + ".txt")
        srt_path = output_dir / (media_file.stem + ".srt")

        print(f"[{i}/{len(todo)}] Transcribing: {media_file.name}")

        segments, info = model.transcribe(
            str(media_file),
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        duration_min = info.duration / 60
        print(f"  Language: {info.language} | Duration: {duration_min:.1f} min")

        all_text = []
        srt_entries = []
        for idx, seg in enumerate(segments, 1):
            all_text.append(seg.text.strip())
            srt_entries.append(
                f"{idx}\n"
                f"{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}\n"
                f"{seg.text.strip()}\n"
            )
            if idx % 20 == 0:
                print(f"  ... {idx} segments done")

        txt_path.write_text("\n".join(all_text), encoding="utf-8")
        srt_path.write_text("\n".join(srt_entries), encoding="utf-8")

        print(f"  [OK] {len(all_text)} segments")
        print(f"    -> {txt_path}")
        print(f"    -> {srt_path}\n")

    print(f"[OK] All transcriptions complete. Output: {output_dir}")


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def list_downloaded(output_dir: Path | None = None) -> None:
    dirs = [output_dir] if output_dir else [DEFAULT_AUDIO_DIR, DEFAULT_VIDEO_DIR]
    found = False
    for d in dirs:
        if not d.exists():
            continue
        media = [f for f in sorted(d.iterdir()) if f.suffix.lstrip(".") in MEDIA_FORMATS]
        if not media:
            continue
        found = True
        print(f"  {d} ({len(media)} files):\n")
        for f in media:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name}  ({size_mb:.1f} MB)")
        print()
    if not found:
        print("No downloaded files found")


def _check_command(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True,
                       encoding="utf-8", errors="replace")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube audio/video download tool (yt-dlp based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --export-cookies
  %(prog)s "URL"
  %(prog)s -f mp3 "URL"
  %(prog)s --video "URL"
  %(prog)s --browser chrome "URL"
  %(prog)s --transcribe
  %(prog)s --transcribe file.m4a
  %(prog)s --transcribe --model small file.m4a
""",
    )
    parser.add_argument("url", nargs="?", help="YouTube video or playlist URL")
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("-f", "--format", default=None)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("-r", "--resolution", default="best")
    parser.add_argument("--browser", choices=["chrome", "firefox", "safari", "edge", "brave", "opera"])
    parser.add_argument("--export-cookies", action="store_true")
    parser.add_argument("--transcribe", nargs="?", const="__all__", default=None, metavar="FILE")
    parser.add_argument("--model", choices=WHISPER_MODELS, default="large-v3")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--list", action="store_true")

    args = parser.parse_args()

    ytdlp = _yt_dlp_cmd()
    if not _check_command(ytdlp):
        print("[ERR] yt-dlp not found. Run: pip install yt-dlp", file=sys.stderr)
        sys.exit(1)

    if args.export_cookies:
        export_cookies(args.browser or "chrome")
        return

    if args.transcribe is not None:
        input_path = None if args.transcribe == "__all__" else Path(args.transcribe)
        transcribe(
            input_path=input_path,
            output_dir=args.output_dir or DEFAULT_TRANSCRIPT_DIR,
            model_size=args.model,
            language=args.lang,
            device=args.device,
        )
        return

    if args.list:
        list_downloaded(args.output_dir)
        return

    if not args.url:
        parser.print_help()
        sys.exit(1)

    is_playlist = "playlist?list=" in args.url

    if is_playlist:
        print("-> Fetching playlist...")
        video_ids = _fetch_playlist_ids(args.url, args.browser)
        print(f"[OK] Playlist has {len(video_ids)} videos\n")
        failed = []
        for i, vid in enumerate(video_ids, 1):
            single_url = f"https://www.youtube.com/watch?v={vid}"
            print(f"{'='*60}")
            print(f"[{i}/{len(video_ids)}] {single_url}")
            print(f"{'='*60}")
            try:
                if args.video:
                    download_video(single_url, args.output_dir or DEFAULT_VIDEO_DIR,
                                   args.format or "mp4", args.resolution, args.browser)
                else:
                    download_audio(single_url, args.output_dir or DEFAULT_AUDIO_DIR,
                                   args.format or "m4a", args.browser)
            except SystemExit:
                print(f"  [WARN] Video {vid} failed, continuing...")
                failed.append(vid)
        print(f"\n[OK] Playlist done: {len(video_ids)-len(failed)}/{len(video_ids)} succeeded")
        if failed:
            print(f"[ERR] Failed: {', '.join(failed)}")
        return

    if args.video:
        fmt = args.format or "mp4"
        if fmt not in VIDEO_FORMATS:
            parser.error(f"Video format must be one of: {', '.join(VIDEO_FORMATS)}")
        download_video(args.url, args.output_dir or DEFAULT_VIDEO_DIR,
                       fmt, args.resolution, args.browser)
    else:
        fmt = args.format or "m4a"
        if fmt not in AUDIO_FORMATS:
            parser.error(f"Audio format must be one of: {', '.join(AUDIO_FORMATS)}")
        download_audio(args.url, args.output_dir or DEFAULT_AUDIO_DIR,
                       fmt, args.browser)


if __name__ == "__main__":
    main()
