#!/usr/bin/env python3
"""Smoke test wrapper for the RhinoFinance daily workflow.

This script intentionally reuses rhino_finance_daily.py instead of keeping a
second copy of the YouTube, LLM, and Feishu logic.
"""

from __future__ import annotations

from datetime import datetime

import rhino_finance_daily as daily


def main() -> None:
    print(f"\n{'=' * 50}")
    print(f"RhinoFinance smoke test - {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'=' * 50}\n")
    print(f"LLM base: {daily.LLM_BASE_URL}")
    print(f"LLM model: {daily.LLM_MODEL}")
    print(f"LLM key configured: {bool(daily.LLM_API_KEY)}")

    video_info = daily.get_today_video()
    if not video_info:
        if daily.LAST_ERROR:
            daily.notify_failure("RhinoFinance video fetch failed", daily.LAST_ERROR)
        return

    video_url, video_title, video_id = video_info
    audio_file = daily.download_audio(video_url, video_id)
    if not audio_file:
        daily.notify_failure(
            "RhinoFinance audio download failed",
            daily.LAST_ERROR or "unknown error",
            video_title,
            video_url,
        )
        return

    transcript = daily.transcribe_audio(audio_file, video_id)
    if not transcript:
        daily.notify_failure(
            "RhinoFinance transcription failed",
            daily.LAST_ERROR or "unknown error",
            video_title,
            video_url,
        )
        return

    summary = daily.summarize_transcript(transcript, video_title)
    if not summary:
        daily.notify_failure(
            "RhinoFinance summary failed",
            daily.LAST_ERROR or "unknown error",
            video_title,
            video_url,
        )
        return

    token = daily.get_tenant_token()
    if token and daily.send_feishu_message(token, video_title, summary, video_url):
        print("RhinoFinance smoke test completed and Feishu message sent.")
    else:
        print("RhinoFinance smoke test completed, but Feishu message was not sent.")


if __name__ == "__main__":
    main()
