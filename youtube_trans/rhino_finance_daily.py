#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RhinoFinance channel daily video download, transcription, and summarization."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
import urllib.request

# ========== Load .env FIRST (before any os.environ.get) ==========
_BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    env_file = _BASE_DIR / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

# ========== Config ==========
BASE_DIR = _BASE_DIR

# Multiple channels: comma-separated pairs "Name|URL,Name2|URL2"
# Falls back to the single-channel env vars for backward compatibility
_CHANNELS_RAW = os.environ.get("RHINO_CHANNELS") or ""
if _CHANNELS_RAW:
    CHANNELS = [
        {"name": p.split("|", 1)[0].strip(), "url": p.split("|", 1)[1].strip()}
        for p in _CHANNELS_RAW.split(",")
        if "|" in p
    ]
else:
    CHANNELS = [
        {
            "name": os.environ.get("RHINO_CHANNEL_NAME") or "RhinoFinance",
            "url": os.environ.get("RHINO_CHANNEL_URL") or "https://www.youtube.com/@RhinoFinance/videos",
        },
        {
            "name": "NaNaShuoMeiGu",
            "url": "https://www.youtube.com/@NaNaShuoMeiGu/videos",
        },
    ]

# Keep single-channel aliases for backward compatibility
CHANNEL_URL = CHANNELS[0]["url"]
CHANNEL_NAME = CHANNELS[0]["name"]
AUDIO_DIR = Path(os.environ.get("RHINO_AUDIO_DIR") or str(Path.home() / "Music" / "yt_audio"))
TRANSCRIPT_DIR = Path(os.environ.get("RHINO_TRANSCRIPT_DIR") or str(Path.home() / "Documents" / "yt_transcripts"))
YT_SCRIPT = Path(os.environ.get("RHINO_YT_SCRIPT") or str(BASE_DIR / "yt_audio.py"))
YT_PYTHON = os.environ.get("RHINO_PYTHON") or sys.executable
YT_DLP = os.environ.get("RHINO_YT_DLP") or "yt-dlp"
MAX_VIDEOS_PER_RUN = int(os.environ.get("RHINO_MAX_VIDEOS_PER_RUN") or "5")
MAX_VIDEO_AGE_DAYS = int(os.environ.get("RHINO_MAX_VIDEO_AGE_DAYS") or "5")
FETCH_RETRIES = int(os.environ.get("RHINO_FETCH_RETRIES") or "3")
RETRY_DELAY = int(os.environ.get("RHINO_RETRY_DELAY") or "10")
# Slow-retry: after all fast retries fail, wait longer and try the whole run again
SLOW_RETRIES = int(os.environ.get("RHINO_SLOW_RETRIES") or "3")
SLOW_RETRY_DELAY = int(os.environ.get("RHINO_SLOW_RETRY_DELAY") or "1800")  # seconds, default 30 min

# Volcano Engine (Ark) LLM config
LLM_BASE_URL = os.environ.get("RHINO_LLM_BASE_URL") or "https://ark.cn-beijing.volces.com/api/coding/v3"
LLM_API_KEY = os.environ.get("ARK_API_KEY") or os.environ.get("RHINO_LLM_API_KEY") or ""
LLM_MODEL = os.environ.get("RHINO_LLM_MODEL") or "minimax-m2.7"

# Feishu push config
FEISHU_APP_ID = os.environ.get("RHINO_FEISHU_APP_ID") or ""
FEISHU_APP_SECRET = os.environ.get("RHINO_FEISHU_APP_SECRET") or ""
FEISHU_USER_OPEN_ID = os.environ.get("RHINO_FEISHU_USER_OPEN_ID") or ""
FEISHU_API = os.environ.get("RHINO_FEISHU_API") or "https://open.feishu.cn/open-apis"

# Proxy (YouTube/volcengine need proxy; Feishu does not)
PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or "http://127.0.0.1:5782"

# Status file tracking processed video IDs
STATUS_FILE = Path(
    os.environ.get("RHINO_STATUS_FILE")
    or str(Path(os.environ.get("TEMP") or "/tmp") / "rhino_finance_status.json")
)

LAST_ERROR = ""

# Log file: same directory as run_daily.bat uses, so ALL invocation paths write here
LOG_FILE = Path(
    os.environ.get("RHINO_LOG_FILE")
    or str(Path(os.environ.get("TEMP") or "/tmp") / "openclaw" / "rhino_finance_daily.log")
)


class _Tee:
    """Write to both the original stream and a log file simultaneously."""

    def __init__(self, stream, log_path: Path):
        self._stream = stream
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = log_path.open("a", encoding="utf-8", buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Proxy other attributes to the underlying stream
    def __getattr__(self, name):
        return getattr(self._stream, name)


def _enable_file_logging() -> None:
    """Redirect stdout/stderr to also write to LOG_FILE (idempotent).

    If the log file is already held open by a parent process (e.g. run_daily.bat
    redirecting stdout), opening it again raises PermissionError — in that case
    we skip installing the Tee since output is already being captured.
    """
    if isinstance(sys.stdout, _Tee):
        return  # already installed
    try:
        tee = _Tee(sys.stdout, LOG_FILE)
    except PermissionError:
        return  # parent already redirects stdout to the log file
    sys.stdout = tee
    sys.stderr = tee


# ========== Helpers ==========

def _require_env(name: str, value: str) -> bool:
    if value:
        return True
    print(f"[ERR] Missing env var: {name}")
    return False


def _has_command(command: str) -> bool:
    return shutil.which(command) is not None


def _resolve_yt_dlp() -> list[str] | None:
    """Return the yt-dlp command list, resolving from venv if needed."""
    if YT_DLP != "yt-dlp":
        return [YT_DLP] if Path(YT_DLP).exists() else None

    python_dir = Path(YT_PYTHON).resolve().parent
    for candidate in [python_dir / "yt-dlp.exe", python_dir / "yt-dlp"]:
        if candidate.exists():
            return [str(candidate)]

    if _has_command("yt-dlp"):
        return ["yt-dlp"]

    return [YT_PYTHON, "-m", "yt_dlp"]


def _preflight_check() -> bool:
    ok = True
    if not YT_SCRIPT.exists():
        print(f"[ERR] yt_audio.py not found: {YT_SCRIPT}")
        ok = False
    if _resolve_yt_dlp() is None:
        print("[ERR] yt-dlp not found. Install: pip install yt-dlp")
        ok = False
    if not _has_command("ffmpeg"):
        print("[ERR] ffmpeg not found. Install: winget install Gyan.FFmpeg")
        ok = False
    return ok


def clear_last_error():
    global LAST_ERROR
    LAST_ERROR = ""


def set_last_error(stage, detail):
    global LAST_ERROR
    LAST_ERROR = f"{stage}: {detail}" if detail else stage


def classify_download_error(detail):
    text = (detail or "").lower()
    if any(k in text for k in ["cookie", "sign in", "login", "members-only",
                                "private video", "premium", "confirm you're not a bot"]):
        return "Cookie or login state may have expired"
    return "Download failed"


def _load_processed_ids():
    try:
        with STATUS_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("processed_ids", []))
    except Exception:
        return set()


def _save_processed_id(video_id):
    try:
        with STATUS_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    ids = set(data.get("processed_ids", []))
    ids.add(video_id)
    data["processed_ids"] = list(ids)[-100:]
    data["last_update"] = datetime.now().isoformat()
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _parse_upload_date(upload_date: str | None):
    if not upload_date or upload_date == "NA":
        return None
    try:
        return datetime.strptime(upload_date, "%Y%m%d")
    except ValueError:
        return None


def _parse_title_date(title: str | None) -> str:
    if not title:
        return ""
    match = re.search(r"(20\d{2})[.\-/](\d{2})[.\-/](\d{2})", title)
    if not match:
        return ""
    return f"{match.group(1)}{match.group(2)}{match.group(3)}"


def _get_video_upload_date(video_url: str) -> str:
    yt_dlp_cmd = _resolve_yt_dlp()
    if yt_dlp_cmd is None:
        return ""

    cmd = [
        *yt_dlp_cmd,
        "--legacy-server-connect",
        "--js-runtimes",
        "node",
        "--skip-download",
        "--print",
        "%(upload_date)s",
        video_url,
    ]
    if PROXY:
        cmd[1:1] = ["--proxy", PROXY]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return ""

    if result.returncode != 0:
        return ""
    if not result.stdout.strip():
        return ""
    value = result.stdout.strip().splitlines()[0].strip()
    return "" if value == "NA" else value


# ========== Step 1: Get latest video ==========

def _fetch_new_videos(channel_url: str, channel_name: str):
    """Fetch unprocessed public videos from a single channel."""
    yt_dlp_cmd = _resolve_yt_dlp()
    if yt_dlp_cmd is None:
        set_last_error("Get video list failed", "yt-dlp not found")
        return []

    cmd = [
        *yt_dlp_cmd,
        "--legacy-server-connect",
        "--flat-playlist",
        "--playlist-end", "15",
        "--print", "id:%(id)s",
        "--print", "title:%(title)s",
        "--print", "upload_date:%(upload_date)s",
        channel_url,
    ]
    if PROXY:
        cmd[1:1] = ["--proxy", PROXY]

    result = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120
            )
        except subprocess.TimeoutExpired:
            if attempt < FETCH_RETRIES:
                delay = RETRY_DELAY * attempt
                print(f"[WARN] [{channel_name}] Fetch timeout (attempt {attempt}/{FETCH_RETRIES}), retrying in {delay}s...")
                time.sleep(delay)
                continue
            print(f"[ERR] yt-dlp timed out for {channel_name} after {FETCH_RETRIES} attempts")
            set_last_error("Get video list failed", "yt-dlp timeout")
            return []

        # Tolerant: SSL warnings may cause non-zero exit but stdout still has data
        if result.returncode != 0 and not result.stdout.strip():
            if attempt < FETCH_RETRIES:
                delay = RETRY_DELAY * attempt
                print(f"[WARN] [{channel_name}] Fetch failed (attempt {attempt}/{FETCH_RETRIES}), retrying in {delay}s...")
                print(f"       stderr: {result.stderr[:200]}")
                time.sleep(delay)
                continue
            print(f"[ERR] yt-dlp failed for {channel_name} after {FETCH_RETRIES} attempts: {result.stderr[:300]}")
            set_last_error("Get video list failed", (result.stderr or result.stdout or "unknown")[:500])
            return []
        break  # Got stdout data (possibly with warnings)

    videos: dict[str, dict] = {}
    current_id = None
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("id:"):
            current_id = line[3:].strip()
            videos[current_id] = {"id": current_id}
        elif line.startswith("title:") and current_id:
            videos[current_id]["title"] = line[6:].strip()
        elif line.startswith("upload_date:") and current_id:
            videos[current_id]["upload_date"] = line[12:].strip()

    if not videos:
        print(f"[WARN] No videos parsed for {channel_name}")
        return []

    _MEMBERS_KEYWORDS = ("会员", "members only", "member only", "members-only")

    processed = _load_processed_ids()
    found = []
    for v in videos.values():
        title = v.get("title", "")
        if any(kw in title.lower() for kw in _MEMBERS_KEYWORDS):
            print(f"[SKIP] [{channel_name}] Members-only video, skipping: {title}")
            continue
        upload_date = v.get("upload_date") or ""
        if upload_date == "NA":
            upload_date = ""
        video_url = f"https://www.youtube.com/watch?v={v['id']}"
        if not upload_date:
            upload_date = _get_video_upload_date(video_url)
        if not upload_date:
            upload_date = _parse_title_date(title)
        uploaded_at = _parse_upload_date(upload_date)
        if uploaded_at and uploaded_at < datetime.now() - timedelta(days=MAX_VIDEO_AGE_DAYS):
            continue
        if v["id"] not in processed:
            found.append(
                {
                    "channel_name": channel_name,
                    "video_url": video_url,
                    "video_title": v.get("title", v["id"]),
                    "video_id": v["id"],
                    "upload_date": upload_date,
                }
            )

    if not found:
        print(f"[INFO] [{channel_name}] All recent videos already processed")
    else:
        print(f"[OK] [{channel_name}] Found {len(found)} unprocessed public videos")
        for item in found:
            print(f"     - {item['video_title']}")

    clear_last_error()
    return found


def get_today_videos():
    """Iterate all channels and return all unprocessed public videos."""
    all_videos = []
    for ch in CHANNELS:
        print(f"[INFO] Searching channel: {ch['name']} ...")
        all_videos.extend(_fetch_new_videos(ch["url"], ch["name"]))
    if all_videos:
        all_videos.sort(key=lambda item: item.get("upload_date", ""), reverse=True)
        print(f"[OK] Total new public videos found: {len(all_videos)}")
        if MAX_VIDEOS_PER_RUN > 0:
            limited = all_videos[:MAX_VIDEOS_PER_RUN]
            print(f"[INFO] Processing up to {len(limited)} videos this run (RHINO_MAX_VIDEOS_PER_RUN={MAX_VIDEOS_PER_RUN})")
            return limited
        return all_videos
    print("[INFO] No new videos found across all channels")
    clear_last_error()
    return []


def process_video(video_info: dict, token: str | None) -> bool:
    video_url = video_info["video_url"]
    video_title = video_info["video_title"]
    video_id = video_info["video_id"]
    channel_name = video_info["channel_name"]

    print(f"\n{'-' * 50}")
    print(f"[INFO] Processing [{channel_name}] {video_title}")
    print(f"[INFO] URL: {video_url}")
    print(f"{'-' * 50}")

    audio_file = download_audio(video_url, video_id)
    if not audio_file:
        print("[ERR] Download audio failed")
        notify_failure("Download audio", LAST_ERROR or "unknown", video_title, video_url)
        return False

    transcript = transcribe_audio(audio_file, video_id)
    if not transcript:
        print("[ERR] Transcription failed")
        notify_failure("Transcription", LAST_ERROR or "unknown", video_title, video_url)
        return False

    print(f"[OK] Transcription done, length: {len(transcript)}")

    summary = summarize_transcript(transcript, video_title)
    if not summary:
        print("[ERR] Summary failed, using transcript excerpt")
        notify_failure("Summary degraded", LAST_ERROR or "LLM failed", video_title, video_url)
        lines = transcript.strip().split("\n")
        summary = "(No AI summary, first 50 lines)\n\n" + "\n".join(lines[:50])

    print("[INFO] Sending Feishu message...")
    try:
        if token:
            if send_feishu_message(token, video_title, summary, video_url):
                print("[OK] Feishu message sent")
            else:
                print("[ERR] Feishu message send failed")
        else:
            print("[ERR] Could not get Feishu token")
    except Exception as e:
        print(f"[ERR] Feishu exception: {e}")

    _save_processed_id(video_id)
    return True


# ========== Step 2: Download audio ==========

def _is_transient_download_error(text: str) -> bool:
    """Return True for network/proxy errors worth retrying."""
    t = text.lower()
    return any(k in t for k in [
        "ssl", "eof", "connection reset", "connection refused", "timed out",
        "timeout", "network", "temporary", "503", "502", "429", "too many requests",
        "read error", "remote end closed",
    ])


def download_audio(video_url, video_id):
    print("[INFO] Downloading audio...")

    existing_files = os.listdir(AUDIO_DIR) if AUDIO_DIR.exists() else []
    audio_file = None
    for f in existing_files:
        if video_id in f and f.endswith(".m4a"):
            audio_file = str(AUDIO_DIR / f)
            print(f"[INFO] Audio already exists: {audio_file}")
            break

    if not audio_file:
        env = dict(os.environ)
        if PROXY:
            env["HTTP_PROXY"] = PROXY
            env["HTTPS_PROXY"] = PROXY
        cmd = [YT_PYTHON, str(YT_SCRIPT), video_url]

        for attempt in range(1, FETCH_RETRIES + 1):
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
            )
            if result.returncode == 0:
                break
            detail = (result.stderr or result.stdout or "unknown")[:1000]
            # Don't retry auth/cookie errors — they won't self-heal
            if not _is_transient_download_error(detail):
                print(f"[ERR] Download failed (non-transient):\n{result.stderr[:500]}")
                set_last_error(classify_download_error(detail), detail)
                return None
            if attempt < FETCH_RETRIES:
                delay = RETRY_DELAY * attempt
                print(f"[WARN] Download failed (attempt {attempt}/{FETCH_RETRIES}), retrying in {delay}s...")
                print(f"       stderr: {result.stderr[:200]}")
                time.sleep(delay)
            else:
                print(f"[ERR] Download failed after {FETCH_RETRIES} attempts:\n{result.stderr[:500]}")
                set_last_error(classify_download_error(detail), detail)
                return None

        if AUDIO_DIR.exists():
            for f in os.listdir(AUDIO_DIR):
                if video_id in f and f.endswith(".m4a"):
                    audio_file = str(AUDIO_DIR / f)
                    break

    if not audio_file:
        set_last_error("Download failed", "Command completed but no .m4a file found")

    return audio_file


# ========== Step 3: Transcribe ==========

def _find_transcript(video_id):
    if not TRANSCRIPT_DIR.is_dir():
        return None
    for name in os.listdir(TRANSCRIPT_DIR):
        if video_id in name and name.endswith(".txt"):
            return str(TRANSCRIPT_DIR / name)
    return None


def transcribe_audio(audio_file, video_id):
    if not audio_file or not os.path.exists(audio_file):
        print("[ERR] Audio file not found")
        set_last_error("Transcription failed", "Audio file not found")
        return None

    existing = _find_transcript(video_id)
    if existing:
        print(f"[INFO] Transcript already exists: {existing}")
        with open(existing, encoding="utf-8") as f:
            return f.read()

    print("[INFO] Transcribing audio...")
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [YT_PYTHON, str(YT_SCRIPT), "--transcribe", "--model", "small", "--device", "cpu", audio_file]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800
    )

    if result.returncode != 0:
        print(f"[ERR] Transcription subprocess failed:\n{result.stderr[:500]}")
        set_last_error("Transcription failed", (result.stderr or result.stdout or "unknown")[:1000])
        return None

    found = _find_transcript(video_id)
    if found:
        with open(found, encoding="utf-8") as f:
            return f.read()

    print("[ERR] Transcription completed but output file not found")
    set_last_error("Transcription failed", "Completed but no transcript file found")
    return None


# ========== Step 4: Summarize with LLM ==========

def summarize_transcript(transcript, video_title):
    print("[INFO] Calling LLM for summary...")

    if not _require_env("ARK_API_KEY or RHINO_LLM_API_KEY", LLM_API_KEY):
        set_last_error("Summary failed", "Ark API Key not configured")
        return None

    summary_prompt = f"""You are a professional financial content analyst. Analyze the following YouTube video transcript and extract key points.

Video title: {video_title}

Transcript:
{transcript[:15000]}

Please respond in Chinese following this exact format:

## \u6838\u5fc3\u89c2\u70b9
1. [\u89c2\u70b91]
2. [\u89c2\u70b92]
3. [\u89c2\u70b93]

## \u5173\u952e\u4e8b\u5b9e/\u6570\u636e
- [\u4e8b\u5b9e1]
- [\u4e8b\u5b9e2]
- [\u4e8b\u5b9e3]
- [\u4e8b\u5b9e4]

## \u7ed3\u8bba\u6216\u5c55\u671b
[\u603b\u7ed3]

Be concise and impactful."""

    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": summary_prompt}],
        "temperature": 0.7,
        "max_tokens": 2000,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
        },
    )

    openers = [urllib.request.build_opener(urllib.request.ProxyHandler({}))]
    if PROXY:
        proxy_handler = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
        openers.append(urllib.request.build_opener(proxy_handler))

    last_exception = None
    for opener in openers:
        try:
            with opener.open(req, timeout=120) as resp:
                result = json.loads(resp.read())
            summary = result["choices"][0]["message"]["content"].strip()
            print("[OK] Summary complete")
            clear_last_error()
            return summary
        except Exception as e:
            last_exception = e

    print(f"[ERR] Summary failed: {last_exception}")
    set_last_error("Summary failed", str(last_exception)[:500])
    return None


# ========== Feishu API ==========
_feishu_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get_tenant_token():
    if not all([
        _require_env("RHINO_FEISHU_APP_ID", FEISHU_APP_ID),
        _require_env("RHINO_FEISHU_APP_SECRET", FEISHU_APP_SECRET),
    ]):
        return None
    url = f"{FEISHU_API}/auth/v3/tenant_access_token/internal"
    data = json.dumps({"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json; charset=utf-8"})
    with _feishu_opener.open(req, timeout=15) as resp:
        result = json.loads(resp.read())
    return result.get("tenant_access_token")


def send_feishu_message(token, title, summary, video_url):
    if not _require_env("RHINO_FEISHU_USER_OPEN_ID", FEISHU_USER_OPEN_ID):
        return False
    url = f"{FEISHU_API}/im/v1/messages?receive_id_type=open_id"

    content_blocks = [
        [{"tag": "text", "text": "Video: "}],
        [{"tag": "a", "href": video_url, "text": video_url}],
        [{"tag": "text", "text": "\n---\n"}],
    ]
    for line in summary.split("\n"):
        line = line.strip()
        if line:
            content_blocks.append([{"tag": "text", "text": line}])

    post_body = {"zh_cn": {"title": f"[Rhino] {title}", "content": content_blocks}}
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {token}",
    }
    data = json.dumps({
        "receive_id": FEISHU_USER_OPEN_ID,
        "msg_type": "post",
        "content": json.dumps(post_body, ensure_ascii=False),
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers)
    with _feishu_opener.open(req, timeout=15) as resp:
        result = json.loads(resp.read())
    return result.get("code") == 0


def send_feishu_alert(token, title, lines):
    if not _require_env("RHINO_FEISHU_USER_OPEN_ID", FEISHU_USER_OPEN_ID):
        return False
    url = f"{FEISHU_API}/im/v1/messages?receive_id_type=open_id"
    content = [[{"tag": "text", "text": line}] for line in lines if line]
    post_body = {"zh_cn": {"title": title, "content": content}}
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {token}",
    }
    data = json.dumps({
        "receive_id": FEISHU_USER_OPEN_ID,
        "msg_type": "post",
        "content": json.dumps(post_body, ensure_ascii=False),
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers)
    with _feishu_opener.open(req, timeout=15) as resp:
        result = json.loads(resp.read())
    return result.get("code") == 0


def notify_failure(stage, detail, video_title=None, video_url=None):
    try:
        token = get_tenant_token()
        if not token:
            print("[ERR] Alert failed: could not get Feishu token")
            return
        lines = [
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Stage: {stage}",
            f"Reason: {(detail or '')[:1200]}",
        ]
        if video_title:
            lines.append(f"Video: {video_title}")
        if video_url:
            lines.append(f"URL: {video_url}")
        if send_feishu_alert(token, "RhinoFinance Task Alert", lines):
            print("[OK] Failure alert sent")
        else:
            print("[ERR] Alert send failed")
    except Exception as e:
        print(f"[ERR] Alert exception: {e}")


# ========== Entry ==========

def main():
    _enable_file_logging()

    if not _preflight_check():
        return

    print(f"\n{'='*50}")
    print(f"Rhino Finance Daily - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Channels: {', '.join(ch['name'] for ch in CHANNELS)}")
    print(f"{'='*50}\n")
    clear_last_error()

    # Slow-retry loop: if proxy is flaky at scheduled time, retry every SLOW_RETRY_DELAY seconds
    for slow_attempt in range(1, SLOW_RETRIES + 1):
        video_list = get_today_videos()
        if video_list:
            break
        # No videos found — could be proxy failure or genuinely nothing new
        if slow_attempt < SLOW_RETRIES:
            delay_min = SLOW_RETRY_DELAY // 60
            print(f"[WARN] No videos found (slow attempt {slow_attempt}/{SLOW_RETRIES}), retrying in {delay_min} min...")
            time.sleep(SLOW_RETRY_DELAY)
            print(f"[INFO] Slow retry {slow_attempt + 1}/{SLOW_RETRIES} starting at {datetime.now().strftime('%H:%M:%S')}")
        else:
            print(f"[INFO] No new videos after {SLOW_RETRIES} slow attempts")

    if not video_list:
        if LAST_ERROR:
            notify_failure("Get video list", LAST_ERROR)
        return

    token = get_tenant_token()
    success_count = 0
    for video_info in video_list:
        if process_video(video_info, token):
            success_count += 1

    print(f"\n{'='*50}")
    print(f"[OK] Task complete! {success_count}/{len(video_list)} videos processed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
