"""
YouTube Transcript tool — fetches transcripts from YouTube videos.
"""

from __future__ import annotations

import logging
import re

try:
    from youtube_transcript_api import YouTubeTranscriptApi

    _HAS_YT = True
except ImportError:
    _HAS_YT = False

from app.tools import register
from app.utils.tool_limiter import check_tool_rate_limit

logger = logging.getLogger(__name__)


@register
def get_youtube_transcript(url: str, lang: str = "en") -> str:
    """
    Fetch the transcript of a YouTube video given its URL or ID.
    Use this when you need to analyze, summarize, or extract information from a video.

    :param url: The YouTube video URL or ID (e.g., 'https://www.youtube.com/watch?v=dQw4w9WgXcQ').
    :param lang: Preferred language code for the transcript (default is 'en').
    """
    if not _HAS_YT:
        return (
            "Error: 'youtube-transcript-api' library not found. Please install it to use this tool."
        )
    if not check_tool_rate_limit("get_youtube_transcript", "10/minute"):
        return "Rate limit exceeded: max 10 YouTube transcript requests per minute."
    try:
        # Extract video ID using regex
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        video_id = video_id_match.group(1) if video_id_match else url
        logger.info(f"Fetching YouTube transcript for ID: {video_id}")
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=[lang])
        full_text = " ".join([t.text for t in transcript])

        return f"--- YouTube Transcript ({video_id}) ---\n{full_text}"

    except Exception as e:
        logger.error(f"YouTube transcript error: {e}")
        return f"Error fetching YouTube transcript: {str(e)}"
