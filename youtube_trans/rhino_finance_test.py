#!/usr/bin/env python3
"""RhinoFinance åæ¬¡æµè¯èæ¬ï¼å¤ç¨ä¸»æµç¨å®ç°ã?""

from __future__ import annotations

from datetime import datetime

import rhino_finance_daily as daily


def main() -> None:
    print(f"\n{'=' * 50}")
    print(f"ð¦ RhinoFinance æµè¯ä»»å¡å¼å§?- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 50}\n")

    video_info = daily.get_today_video()
    if not video_info:
        if daily.LAST_ERROR:
            daily.notify_failure("è·åè§é¢åè¡¨", daily.LAST_ERROR)
        return

    video_url, video_title, video_id = video_info
    audio_file = daily.download_audio(video_url, video_id)
    if not audio_file:
        daily.notify_failure("ä¸è½½é³é¢", daily.LAST_ERROR or "æªç¥éè¯¯", video_title, video_url)
        return

    transcript = daily.transcribe_audio(audio_file, video_id)
    if not transcript:
        daily.notify_failure("è½¬å", daily.LAST_ERROR or "æªç¥éè¯¯", video_title, video_url)
        return

    summary = daily.summarize_transcript(transcript, video_title)
    if not summary:
        daily.notify_failure("æ»ç»", daily.LAST_ERROR or "æªç¥éè¯¯", video_title, video_url)
        return

    token = daily.get_tenant_token()
    if token and daily.send_feishu_message(token, video_title, summary, video_url):
        print("â?æµè¯æ¶æ¯åéæå?)
    else:
        print("â?æµè¯æ¶æ¯åéå¤±è´?)


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
RhinoFinance é¢éæµè¯èæ¬ â?ä¸è½½æè¿ä¸ä¸ªè§é¢æµè¯å®æ´æµç¨?"""

import subprocess
import os
import sys
from datetime import datetime, timedelta
import json
import re
import urllib.parse
import urllib.request

# ========== éç½® ==========
CHANNEL_URL = "https://www.youtube.com/@RhinoFinance/videos"
AUDIO_DIR = os.path.expanduser("~/Music/yt_audio")
TRANSCRIPT_DIR = os.path.expanduser("~/Documents/yt_transcripts")
YT_SCRIPT = "/Users/wl/gem_claude/yt_audio.py"
YT_PYTHON = "/Users/wl/.pyenv/shims/python"
CHANNEL_NAME = "RhinoFinance"

# ç«å±±å¼æ API éç½®
LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
LLM_API_KEY = "34081167-83fa-43c5-9c30-632e640fba9c"
LLM_MODEL = "ark-code-latest"

# é£ä¹¦æ¶æ¯æ¨ééç½?FEISHU_APP_ID = "cli_a924285ae7f85cc7"
FEISHU_APP_SECRET = "53hrIbxJYHGGAI8qbndwofOzltJAkah0"
FEISHU_USER_OPEN_ID = "ou_5489407346c5c13bc4687a83859d619b"
FEISHU_API = "https://open.feishu.cn/open-apis"

# ä»£çéç½®
PROXY = os.environ.get("HTTPS_PROXY", os.environ.get("HTTP_PROXY", "http://127.0.0.1:5782"))

def get_latest_video():
    """è·åé¢éæè¿çä¸ä¸ªè§é¢ï¼ä¸æ¯åªè·åå½å¤©çï¼?""
    print(f"ð æ¥æ¾ {CHANNEL_NAME} é¢éæè¿çè§é¢...")

    cmd = [
        "yt-dlp",
    ]
    if PROXY:
        cmd.extend(["--proxy", PROXY])
    cmd.extend([
        "--flat-playlist",
        "--playlist-end", "1",
        "--format", "best",
        "--print", "id:%(id)s",
        "--print", "title:%(title)s",
        "--print", "upload_date:%(upload_date)s",
        CHANNEL_URL
    ])

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if result.returncode != 0:
        print(f"â?è·åè§é¢åè¡¨å¤±è´¥: {result.stderr}")
        return None

    videos = {}
    current_id = None
    for line in result.stdout.strip().split('\n'):
        if line.startswith('id:'):
            current_id = line[3:]
            videos[current_id] = {'id': current_id}
        elif line.startswith('title:') and current_id:
            videos[current_id]['title'] = line[6:]
        elif line.startswith('upload_date:') and current_id:
            videos[current_id]['upload_date'] = line[12:]

    if not videos:
        print(f"â?æ²¡ææ¾å°è§é¢")
        return None

    latest = list(videos.values())[0]
    video_url = f"https://www.youtube.com/watch?v={latest['id']}"
    print(f"â?æ¾å°ææ°è§é¢? {latest['title']}")
    print(f"   ä¸ä¼ æ¥æ: {latest.get('upload_date', 'unknown')}")
    print(f"   URL: {video_url}")

    return video_url, latest['title'], latest['id']

def download_audio(video_url, video_id):
    """ä¸è½½è§é¢é³é¢"""
    print(f"â¬ï¸ ä¸è½½é³é¢ä¸?..")

    existing_files = os.listdir(AUDIO_DIR) if os.path.exists(AUDIO_DIR) else []
    audio_file = None
    for f in existing_files:
        if video_id in f and f.endswith('.m4a'):
            audio_file = os.path.join(AUDIO_DIR, f)
            print(f"â¹ï¸ é³é¢å·²å­å? {audio_file}")
            break

    if not audio_file:
        # åè®¾ç½®ä»£çç¯å¢åéï¼ç¶åè°ç¨èæ¬
        env = dict(os.environ)
        if PROXY:
            env["HTTP_PROXY"] = PROXY
            env["HTTPS_PROXY"] = PROXY
        cmd = [
            YT_PYTHON, YT_SCRIPT,
            "--browser", "chrome",
            video_url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)

        if result.returncode != 0:
            print(f"â?ä¸è½½å¤±è´¥: {result.stderr}")
            return None

        for f in os.listdir(AUDIO_DIR):
            if video_id in f and f.endswith('.m4a'):
                audio_file = os.path.join(AUDIO_DIR, f)
                break

    return audio_file

def transcribe_audio(audio_file, video_id):
    """è½¬åé³é¢"""
    if not audio_file or not os.path.exists(audio_file):
        print(f"â?é³é¢æä»¶ä¸å­å?)
        return None

    transcript_file = os.path.join(TRANSCRIPT_DIR, f"{video_id}.txt")
    if os.path.exists(transcript_file):
        print(f"â¹ï¸ è½¬åæä»¶å·²å­å? {transcript_file}")
        with open(transcript_file, 'r', encoding='utf-8') as f:
            return f.read()

    print(f"ð è½¬åé³é¢ä¸?..")
    cmd = [
        YT_PYTHON, YT_SCRIPT,
        "--transcribe",
        "--model", "small",
        audio_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if result.returncode != 0:
        print(f"â?è½¬åå¤±è´¥: {result.stderr}")
        return None

    if os.path.exists(transcript_file):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            return f.read()

    return None

def summarize_transcript(transcript, video_title):
    """ä½¿ç¨å¤§æ¨¡åæ»ç»è½¬ååå®¹"""
    print(f"ð è°ç¨å¤§æ¨¡åæ»ç»ä¸?..")

    # æå»º prompt
    summary_prompt = f"""ä½ æ¯ä¸ä¸ªä¸ä¸çéèåå®¹åæå¸ãè¯·åæä»¥ä¸ YouTube è§é¢è½¬ååå®¹ï¼æåæ ¸å¿è§ç¹åå³é®äºå®ã?
è§é¢æ é¢: {video_title}

è½¬ååå®¹:
{transcript[:15000]}

è¯·ä¸¥æ ¼æç§ä»¥ä¸æ ¼å¼æ»ç»ï¼?
## æ ¸å¿è§ç¹
1. [è§ç¹1]
2. [è§ç¹2]
3. [è§ç¹3]

## å³é®äºå®/æ°æ®
- [äºå®1]
- [äºå®2]
- [äºå®3]
- [äºå®4]

## ç»è®ºæå±æ?[æ»ç»æ§æè¿°]

è¯·ç¨ä¸­æåå¤ï¼è¯­è¨ç®æ´æåã?""

    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": summary_prompt}],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        summary = result["choices"][0]["message"]["content"].strip()
        print(f"â?æ»ç»å®æ")
        return summary
    except Exception as e:
        print(f"â?æ»ç»å¤±è´¥: {e}")

    return None

# ========== é£ä¹¦ API ==========
def get_tenant_token():
    """è·åé£ä¹¦ tenant_access_token"""
    url = f"{FEISHU_API}/auth/v3/tenant_access_token/internal"
    headers = {"Content-Type": "application/json; charset=utf-8"}
    data = json.dumps({
        "app_id": FEISHU_APP_ID,
        "app_secret": FEISHU_APP_SECRET,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read())
    return result.get("tenant_access_token")

def send_feishu_message(token, title, summary, video_url):
    """åéé£ä¹¦å¯ææ¬æ¶æ¯"""
    url = f"{FEISHU_API}/im/v1/messages?receive_id_type=open_id"

    # æå»ºå¯ææ¬åå®?    content_blocks = []
    content_blocks.append([{"tag": "text", "text": "ð è§é¢é¾æ¥: "}])
    content_blocks.append([{"tag": "a", "href": video_url, "text": video_url}])
    content_blocks.append([{"tag": "text", "text": "\n---\n"}])

    # éè¡æ·»å æ»ç»åå®¹
    for line in summary.split('\n'):
        line = line.strip()
        if line:
            content_blocks.append([{"tag": "text", "text": line}])

    post_body = {"zh_cn": {"title": f"ð¦ {title}", "content": content_blocks}}

    headers = {"Content-Type": "application/json; charset=utf-8"}
    headers["Authorization"] = f"Bearer {token}"

    data = json.dumps({
        "receive_id": FEISHU_USER_OPEN_ID,
        "msg_type": "post",
        "content": json.dumps(post_body, ensure_ascii=False),
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read())
    return result.get("code") == 0

def main():
    print(f"\n{'='*50}")
    print(f"ð¦ RhinoFinance æµè¯ä»»å¡å¼å§?- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

    # 1. è·åæè¿çä¸ä¸ªè§é¢?    video_info = get_latest_video()
    if not video_info:
        print(f"â?æ²¡ææ¾å°è§é¢")
        return

    video_url, video_title, video_id = video_info

    # 2. ä¸è½½é³é¢
    audio_file = download_audio(video_url, video_id)
    if not audio_file:
        print(f"â?ä¸è½½é³é¢å¤±è´¥")
        return

    # 3. è½¬å
    transcript = transcribe_audio(audio_file, video_id)
    if not transcript:
        print(f"â?è½¬åå¤±è´¥")
        return

    print(f"â?è½¬åå®æï¼å­æ? {len(transcript)}")

    # 4. æ»ç»
    summary = summarize_transcript(transcript, video_title)
    if not summary:
        print(f"â?æ»ç»å¤±è´¥ï¼ä½¿ç¨ç®åæè¦?)
        lines = transcript.strip().split('\n')
        summary = "ï¼æ  AI æ»ç»ï¼ä»æ¾ç¤ºè½¬åå?50 è¡ï¼\n\n" + '\n'.join(lines[:50])

    # 5. åéé£ä¹¦æ¶æ?    print(f"ð¤ åéé£ä¹¦æ¶æ?..")
    try:
        token = get_tenant_token()
        if token:
            if send_feishu_message(token, video_title, summary, video_url):
                print(f"â?é£ä¹¦æ¶æ¯åéæå?)
            else:
                print(f"â?é£ä¹¦æ¶æ¯åéå¤±è´?)
        else:
            print(f"â?è·åé£ä¹¦ token å¤±è´¥")
    except Exception as e:
        print(f"â?é£ä¹¦æ¶æ¯åéå¼å¸? {e}")

    # è¾åºç»æ
    print(f"\n{'='*50}")
    print(f"â?æµè¯ä»»å¡å®æ!")
    print(f"{'='*50}")

    print(f"\n{summary}")

    # ä¿å­ç¶æ?    status = {
        "status": "success",
        "date": datetime.now().isoformat(),
        "video_title": video_title,
        "video_url": video_url,
        "video_id": video_id,
        "transcript_length": len(transcript),
        "summary": summary
    }
    with open("/tmp/rhino_finance_test_status.json", "w") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    return status

if __name__ == "__main__":
    main()
